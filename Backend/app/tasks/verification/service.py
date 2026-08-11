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


@dataclass(frozen=True, slots=True)
class SpeakerModelSpec:
    key: str
    label: str
    model_id: str
    architecture: str
    embedding_dimension: int
    threshold: float
    recommended: bool


MODEL_SPECS: dict[str, SpeakerModelSpec] = {
    "ecapa-tdnn": SpeakerModelSpec(
        key="ecapa-tdnn",
        label="ECAPA-TDNN",
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        architecture="ECAPA-TDNN",
        embedding_dimension=192,
        threshold=0.3578562438488006,
        recommended=True,
    ),
    "resnet34-lm": SpeakerModelSpec(
        key="resnet34-lm",
        label="WeSpeaker ResNet34-LM",
        model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
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
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as error:
            raise SpeakerModelUnavailable(
                "ECAPA-TDNN requires the 'speechbrain' package. "
                "Install the backend requirements and restart the API."
            ) from error

        try:
            self.model = EncoderClassifier.from_hparams(
                source=spec.model_id,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
                local_strategy=LocalStrategy.COPY,
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

        previous_setting = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            model = Model.from_pretrained(spec.model_id)
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


def temporal_occlusion_saliency(
    model_key: str,
    enrollment_paths: Sequence[str | Path],
    probe_path: str | Path,
    segment_count: int = 8,
) -> dict[str, object]:
    """Measure score change after muting each temporal region of the probe."""

    if not 3 <= len(enrollment_paths) <= 5:
        raise ValueError("Speaker enrolment requires between 3 and 5 reference recordings.")
    if not 4 <= segment_count <= 20:
        raise ValueError("Temporal occlusion requires between 4 and 20 segments.")

    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    enrollment_embeddings = [adapter.extract_embedding(path) for path in enrollment_paths]
    centroid = F.normalize(torch.stack(enrollment_embeddings).mean(dim=0), p=2, dim=0)
    baseline_embedding = adapter.extract_embedding(probe_path)
    baseline_similarity = _cosine(centroid, baseline_embedding)

    waveform, sample_rate = torchaudio.load(str(probe_path))
    if waveform.numel() == 0:
        raise ValueError("Probe audio is empty.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_samples = waveform.shape[1]
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
            segments.append(
                {
                    "segment_index": index + 1,
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                    "occluded_similarity": occluded_similarity,
                    "importance": baseline_similarity - occluded_similarity,
                }
            )

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