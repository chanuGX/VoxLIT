"""Speaker-verification model loading and inference.

The two production models deliberately stay behind the same small adapter
contract while retaining separate caches, thresholds, and embedding spaces.
Heavy optional dependencies are imported only when a model is first used so
the rest of the VoxLIT API can still start without downloading SV weights.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from app.core.settings import settings
from app.tasks.verification import clustering


@dataclass(frozen=True, slots=True)
class SpeakerModelSpec:
    key: str
    label: str
    model_id: str
    revision: str
    architecture: str
    embedding_dimension: int
    threshold: float
    recommended: bool


# Pair-verification thresholds are EER-calibrated and locked before test
# evaluation (see the `calibration` block returned by `verify_speaker`).
# This version identifies that calibration methodology for cache keys.
PAIR_THRESHOLD_VERSION = "pair-threshold-v1-eer"

# Identifies the fixed preprocessing pipeline in `_BaseAdapter`: 16kHz
# resample, mono downmix, L2-normalized output embedding.
PREPROCESSING_VERSION = "verification-preprocess-v1"

MODEL_SPECS: dict[str, SpeakerModelSpec] = {
    "ecapa-tdnn": SpeakerModelSpec(
        key="ecapa-tdnn",
        label="ECAPA-TDNN",
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        architecture="ECAPA-TDNN",
        embedding_dimension=192,
        threshold=0.3578562438488006,
        recommended=True,
    ),
    "resnet34-lm": SpeakerModelSpec(
        key="resnet34-lm",
        label="WeSpeaker ResNet34-LM",
        model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        revision="837717ddb9ff5507820346191109dc79c958d614",
        architecture="ResNet34-LM",
        embedding_dimension=256,
        threshold=0.3843278586864471,
        recommended=False,
    ),
}


class SpeakerEmbeddingAdapter(Protocol):
    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor: ...


class UnsupportedSpeakerModel(ValueError):
    """Raised when the API receives a model key outside the task registry."""


class SpeakerModelUnavailable(RuntimeError):
    """Raised when an optional model dependency or checkpoint cannot load."""


class _BaseAdapter:
    target_sample_rate = 16_000

    def __init__(self, expected_dimension: int) -> None:
        self.expected_dimension = expected_dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Audio contains invalid values: {audio_path}")
        return waveform

    def validate_embedding(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        embedding = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if embedding.numel() != self.expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.expected_dimension}, got {embedding.numel()}"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError("Embedding contains non-finite values.")
        if float(torch.linalg.vector_norm(embedding)) == 0.0:
            raise ValueError("Embedding has zero magnitude.")
        return F.normalize(embedding, p=2, dim=0).detach().cpu()


class _ECAPAAdapter(_BaseAdapter):
    def __init__(self, spec: SpeakerModelSpec) -> None:
        super().__init__(spec.embedding_dimension)
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import FetchConfig, LocalStrategy
        except ImportError as error:
            raise SpeakerModelUnavailable(
                "ECAPA-TDNN requires the 'speechbrain' package. "
                "Install the backend requirements and restart the API."
            ) from error

        savedir = settings.speaker_verification_ecapa_dir_for_revision(spec.revision)
        hf_cache_dir = settings.speaker_verification_hf_cache_dir
        savedir.mkdir(parents=True, exist_ok=True)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.model = EncoderClassifier.from_hparams(
                source=spec.model_id,
                savedir=str(savedir),
                run_opts={"device": str(self.device)},
                local_strategy=LocalStrategy.COPY,
                fetch_config=FetchConfig(
                    revision=spec.revision,
                    huggingface_cache_dir=str(hf_cache_dir),
                ),
            )
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as error:
            raise SpeakerModelUnavailable(f"Could not load ECAPA-TDNN: {error}") from error

    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        relative_length = torch.tensor([1.0], device=self.device)
        with torch.inference_mode():
            embedding = self.model.encode_batch(
                waveform,
                wav_lens=relative_length,
                normalize=False,
            )
        return self.validate_embedding(embedding)


class _WeSpeakerAdapter(_BaseAdapter):
    def __init__(self, spec: SpeakerModelSpec) -> None:
        super().__init__(spec.embedding_dimension)
        try:
            from pyannote.audio import Inference, Model
        except ImportError as error:
            raise SpeakerModelUnavailable(
                "ResNet34-LM requires the 'pyannote.audio' package. "
                "Install the backend requirements and restart the API."
            ) from error

        hf_cache_dir = settings.speaker_verification_hf_cache_dir
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        previous_setting = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            checkpoint = f"{spec.model_id}@{spec.revision}"
            model = Model.from_pretrained(checkpoint, cache_dir=str(hf_cache_dir))
        except Exception as error:
            raise SpeakerModelUnavailable(f"Could not load ResNet34-LM: {error}") from error
        finally:
            if previous_setting is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_setting

        if model is None:
            raise SpeakerModelUnavailable(f"Could not load ResNet34-LM: {spec.model_id}")
        model.eval()
        self.inference = Inference(model, window="whole")
        self.inference.to(self.device)

    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        with torch.inference_mode():
            embedding = self.inference(
                {"waveform": waveform, "sample_rate": self.target_sample_rate}
            )
        return self.validate_embedding(np.asarray(embedding, dtype=np.float32))


_MODEL_CACHE: dict[str, SpeakerEmbeddingAdapter] = {}
_LOAD_LOCK = threading.Lock()


def list_models() -> list[dict[str, object]]:
    """Return public metadata for the two models selected by evaluation."""

    return [asdict(spec) for spec in MODEL_SPECS.values()]


def get_model_spec(model_key: str) -> SpeakerModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        valid = ", ".join(MODEL_SPECS)
        raise UnsupportedSpeakerModel(
            f"Unsupported speaker-verification model '{model_key}'. Valid models: {valid}."
        ) from error


def get_model(model_key: str) -> SpeakerEmbeddingAdapter:
    """Load a selected model once; concurrent first requests cannot race."""

    spec = get_model_spec(model_key)
    with _LOAD_LOCK:
        if model_key not in _MODEL_CACHE:
            if model_key == "ecapa-tdnn":
                _MODEL_CACHE[model_key] = _ECAPAAdapter(spec)
            else:
                _MODEL_CACHE[model_key] = _WeSpeakerAdapter(spec)
        return _MODEL_CACHE[model_key]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())


def _compactness(embeddings: Sequence[torch.Tensor]) -> float:
    scores = [_cosine(a, b) for a, b in combinations(embeddings, 2)]
    return float(np.mean(scores)) if scores else 1.0


def verify_speaker(
    model_key: str,
    enrollment_paths: Sequence[str | Path],
    probe_path: str | Path,
) -> dict[str, object]:
    """Compare a probe with the L2-normalised centroid of 3–5 references."""

    if not 3 <= len(enrollment_paths) <= 5:
        raise ValueError("Speaker enrolment requires between 3 and 5 reference recordings.")

    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    enrollment_embeddings = [adapter.extract_embedding(path) for path in enrollment_paths]
    probe_embedding = adapter.extract_embedding(probe_path)

    centroid = torch.stack(enrollment_embeddings).mean(dim=0)
    centroid = F.normalize(centroid, p=2, dim=0)
    similarity = _cosine(centroid, probe_embedding)
    per_reference_scores = [
        {
            "reference_index": index + 1,
            "similarity": _cosine(reference, probe_embedding),
        }
        for index, reference in enumerate(enrollment_embeddings)
    ]

    return {
        "model": spec.key,
        "model_label": spec.label,
        "model_id": spec.model_id,
        "embedding_dimension": spec.embedding_dimension,
        "scoring_method": "cosine_similarity",
        "enrollment_count": len(enrollment_paths),
        "similarity": similarity,
        "threshold": spec.threshold,
        "decision_margin": similarity - spec.threshold,
        "same_speaker": similarity >= spec.threshold,
        "enrollment_compactness": _compactness(enrollment_embeddings),
        "per_reference_scores": per_reference_scores,
        "enrollment_centroid": centroid.tolist(),
        "probe_embedding": probe_embedding.tolist(),
        "calibration": {
            "criterion": "equal_error_rate",
            "dataset": "VoxCeleb1 Indian verification subset",
            "threshold_locked_before_test_evaluation": True,
        },
    }


def extract_batch_embeddings(
    model_key: str, audio_paths: Sequence[str | Path]
) -> torch.Tensor:
    """Extract one L2-normalised embedding per recording, in request order."""

    adapter = get_model(model_key)
    embeddings = [adapter.extract_embedding(path) for path in audio_paths]
    return torch.stack(embeddings)


def pairwise_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """Symmetric NxN cosine-similarity matrix with an exact unit diagonal."""

    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = (normalized @ normalized.T).detach().cpu().numpy()
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity = (similarity + similarity.T) / 2.0
    np.fill_diagonal(similarity, 1.0)
    return similarity


def pairwise_decision_matrix(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Symmetric NxN same-speaker decision matrix from a similarity matrix."""

    decisions = similarity >= threshold
    np.fill_diagonal(decisions, True)
    return decisions


def assemble_batch_analysis(
    model_key: str,
    embeddings: torch.Tensor,
    labels: Sequence[str],
) -> dict[str, object]:
    """Pairwise similarity/decision/clustering analysis from precomputed embeddings.

    Split out of `batch_verification_analysis` so a caller can supply a
    tensor merged from cached and freshly extracted embeddings without
    re-running extraction.
    """

    if embeddings.shape[0] != len(labels):
        raise ValueError("Embeddings and labels must have the same length.")

    spec = get_model_spec(model_key)
    similarity = pairwise_similarity_matrix(embeddings)
    decisions = pairwise_decision_matrix(similarity, spec.threshold)
    cluster_result = clustering.cluster_batch(model_key, similarity, labels)

    return {
        "model": spec.key,
        "model_label": spec.label,
        "threshold": spec.threshold,
        "recording_count": len(labels),
        "embedding_dimension": spec.embedding_dimension,
        "labels": list(labels),
        "embeddings": embeddings.tolist(),
        "similarity_matrix": similarity.tolist(),
        "decision_matrix": decisions.tolist(),
        **cluster_result,
    }


def batch_verification_analysis(
    model_key: str,
    audio_paths: Sequence[str | Path],
    labels: Sequence[str],
) -> dict[str, object]:
    """Pairwise embedding/similarity/decision analysis for 2-100 recordings."""

    if not 2 <= len(audio_paths) <= 100:
        raise ValueError("Batch verification requires between 2 and 100 recordings.")
    if len(audio_paths) != len(labels):
        raise ValueError("Recording paths and labels must have the same length.")

    embeddings = extract_batch_embeddings(model_key, audio_paths)
    return assemble_batch_analysis(model_key, embeddings, labels)


def _validate_embedding_matrix(
    embeddings: Sequence[Sequence[float]], expected_dimension: int
) -> np.ndarray:
    """Build a float64 matrix, rejecting shapes/values a projection can't use.

    A single dimension check covers empty rows, ragged rows, and oversized
    rows: every row must match the model's exact production embedding
    dimension, nothing else is a valid speaker-verification embedding.
    """

    row_lengths = {len(row) for row in embeddings}
    if row_lengths != {expected_dimension}:
        raise ValueError(
            f"Each embedding must have exactly {expected_dimension} dimensions "
            f"(found length(s): {sorted(row_lengths)})."
        )

    matrix = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must not contain NaN or infinite values.")
    return matrix


def project_batch_embeddings(
    embeddings: Sequence[Sequence[float]],
    model_key: str,
    reduction_method: str,
    n_components: int,
) -> tuple[np.ndarray, int, str]:
    """Visualization-only PCA/t-SNE/UMAP projection of precomputed embeddings.

    Never used for clustering — clustering already runs on raw embeddings in
    `assemble_batch_analysis`/`clustering.cluster_batch`. Always returns
    exactly `n_components` columns: PCA (and PCA-as-UMAP-fallback) cannot
    manufacture more components than a very small batch allows (e.g. 2
    recordings requesting 3D), so the shortfall is deterministically
    zero-padded and the achievable count is reported separately. t-SNE and
    UMAP never need padding here — for tiny batches they run with
    `init="random"` instead of their density/spectral-based default init,
    which removes their sample-count ceiling entirely (verified empirically
    against sklearn 1.9 / umap-learn 0.5.12: the default `init="pca"` for
    t-SNE and `init="spectral"` for UMAP both fail below a measured sample
    threshold, independent of any `perplexity`/`n_neighbors` clamping).

    Returns (coordinates, effective_components, reduction_method_used).
    """

    spec = get_model_spec(model_key)
    matrix = _validate_embedding_matrix(embeddings, spec.embedding_dimension)
    n_samples, n_features = matrix.shape

    method_used = reduction_method
    if reduction_method == "umap" and n_samples < 3:
        # UMAP requires n_neighbors >= 2, which requires >= 3 samples under
        # any initialization strategy — deterministic, documented fallback.
        method_used = "pca"

    try:
        if method_used == "pca":
            from sklearn.decomposition import PCA

            achievable = min(n_samples, n_features, n_components)
            coords = PCA(n_components=achievable, random_state=42).fit_transform(matrix)
        elif method_used == "tsne":
            from sklearn.manifold import TSNE

            perplexity = min(30, n_samples - 1)
            init = "random" if n_samples < n_components else "pca"
            coords = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=perplexity,
                init=init,
            ).fit_transform(matrix)
            achievable = n_components
        else:  # umap, n_samples >= 3
            import umap

            n_neighbors = max(2, min(15, n_samples - 1))
            init = "random" if n_samples < n_components + 2 else "spectral"
            coords = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=n_neighbors,
                init=init,
            ).fit_transform(matrix)
            achievable = n_components
    except Exception as error:
        # Normalize any library-internal exception (e.g. UMAP's TypeError on
        # spectral init for small batches) into a ValueError, so callers get
        # one consistent failure type regardless of which reducer ran.
        raise ValueError(f"Projection failed: {error}") from error

    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords, min(achievable, n_components), method_used


def rehydrate_cached_embedding(
    model_key: str, values: Sequence[float]
) -> torch.Tensor | None:
    """Validate and L2-renormalize a cache-hit embedding, or None if invalid.

    Mirrors `_BaseAdapter.validate_embedding`'s checks (dimension, finite,
    non-zero magnitude) but returns None instead of raising, so an invalid
    cached entry is treated as a cache miss rather than an error.
    """

    spec = get_model_spec(model_key)
    try:
        tensor = torch.tensor(values, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if tensor.numel() != spec.embedding_dimension:
        return None
    if not torch.isfinite(tensor).all():
        return None
    if float(torch.linalg.vector_norm(tensor)) == 0.0:
        return None
    return F.normalize(tensor, p=2, dim=0)


def _occlude_and_score(
    adapter: SpeakerEmbeddingAdapter,
    centroid: torch.Tensor,
    target_path: str | Path,
    segment_count: int,
) -> tuple[float, float, list[dict[str, float | int]]]:
    """Mute each temporal region of target_path once, scoring against a fixed
    centroid. Returns (baseline_similarity, audio_duration_seconds, segments).

    No reference-count or segment-count constraints here -- callers own those
    (they differ between the 3-5-enrollment pair-verification path and the
    unbounded-membership cluster path)."""

    baseline_embedding = adapter.extract_embedding(target_path)
    baseline_similarity = _cosine(centroid, baseline_embedding)

    waveform, sample_rate = torchaudio.load(str(target_path))
    if waveform.numel() == 0:
        raise ValueError("Target audio is empty.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_samples = waveform.shape[1]
    audio_duration_seconds = total_samples / sample_rate
    boundaries = np.linspace(0, total_samples, segment_count + 1, dtype=int)
    segments: list[dict[str, float | int]] = []

    with TemporaryDirectory(prefix="voxlit-occlusion-") as temp_dir:
        for index in range(segment_count):
            start_sample = int(boundaries[index])
            end_sample = int(boundaries[index + 1])
            occluded = waveform.clone()
            occluded[:, start_sample:end_sample] = 0.0
            occluded_path = Path(temp_dir) / f"segment-{index}.wav"
            torchaudio.save(str(occluded_path), occluded, sample_rate)

            occluded_embedding = adapter.extract_embedding(occluded_path)
            occluded_similarity = _cosine(centroid, occluded_embedding)
            similarity_change = baseline_similarity - occluded_similarity
            segments.append(
                {
                    "segment_index": index + 1,
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                    "occluded_similarity": occluded_similarity,
                    "similarity_change": similarity_change,
                    "influence_strength": abs(similarity_change),
                }
            )

    return baseline_similarity, audio_duration_seconds, segments


def temporal_occlusion_saliency(
    model_key: str,
    enrollment_paths: Sequence[str | Path],
    probe_path: str | Path,
    segment_count: int = 8,
) -> dict[str, object]:
    """Measure score change after muting each temporal region of the probe.

    Public response shape is unchanged from before the `_occlude_and_score`
    refactor -- this wrapper still owns the 3-5-enrollment validation and
    maps each segment back to the original `importance`-only shape."""

    if not 3 <= len(enrollment_paths) <= 5:
        raise ValueError("Speaker enrolment requires between 3 and 5 reference recordings.")
    if not 4 <= segment_count <= 20:
        raise ValueError("Temporal occlusion requires between 4 and 20 segments.")

    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    enrollment_embeddings = [adapter.extract_embedding(path) for path in enrollment_paths]
    centroid = F.normalize(torch.stack(enrollment_embeddings).mean(dim=0), p=2, dim=0)

    baseline_similarity, _audio_duration_seconds, raw_segments = _occlude_and_score(
        adapter, centroid, probe_path, segment_count
    )
    segments = [
        {
            "segment_index": segment["segment_index"],
            "start_seconds": segment["start_seconds"],
            "end_seconds": segment["end_seconds"],
            "occluded_similarity": segment["occluded_similarity"],
            "importance": segment["similarity_change"],
        }
        for segment in raw_segments
    ]

    return {
        "model": spec.key,
        "model_label": spec.label,
        "baseline_similarity": baseline_similarity,
        "threshold": spec.threshold,
        "segment_count": segment_count,
        "interpretation": (
            "Positive importance means the muted segment supported the original "
            "speaker-similarity score; negative importance means it opposed it."
        ),
        "segments": segments,
    }


def compute_saliency_map(
    model_key: str,
    reference_paths: Sequence[str | Path],
    target_path: str | Path,
    *,
    reference_type: str,
    cluster_id: str | None = None,
    target_recording_id: str | None = None,
    segment_count: int = 8,
) -> dict[str, object]:
    """Generalized temporal-occlusion saliency: fixed reference centroid,
    single target, no reclustering, no upper/lower bound tied to the 3-5
    pair-verification enrollment rule.

    `reference_type`/`cluster_id`/`target_recording_id` are pure passthrough
    metadata the caller already knows (this function never derives or
    validates them against any stored clustering -- there is none). Reference
    and segment-count bounds are the router's job (they differ between
    enrollment and cluster mode); this function only rejects an empty
    reference list and an out-of-range segment count so it stays safe to
    call directly (e.g. from tests) without going through the router.
    """

    if not reference_paths:
        raise ValueError("At least one reference recording is required.")
    if not 4 <= segment_count <= 20:
        raise ValueError("Temporal occlusion requires between 4 and 20 segments.")

    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    reference_embeddings = [adapter.extract_embedding(path) for path in reference_paths]
    centroid = F.normalize(torch.stack(reference_embeddings).mean(dim=0), p=2, dim=0)

    baseline_similarity, audio_duration_seconds, segments = _occlude_and_score(
        adapter, centroid, target_path, segment_count
    )

    return {
        "model": spec.key,
        "model_label": spec.label,
        "reference_type": reference_type,
        "cluster_id": cluster_id,
        "target_recording_id": target_recording_id,
        "reference_count": len(reference_paths),
        "baseline_similarity": baseline_similarity,
        "threshold": spec.threshold,
        "segment_count": segment_count,
        "audio_duration_seconds": audio_duration_seconds,
        "interpretation": (
            "Positive similarity_change means the muted segment supported the "
            "match to the reference centroid; negative similarity_change means "
            "it opposed it. influence_strength is the absolute value, used for "
            "ranking segments by influence regardless of direction."
        ),
        "segments": segments,
    }