"""Audio Deepfake Detection — model loading and inference.

Model A is a Tier A checkpoint: it ships `config.json` +
`preprocessor_config.json`, so it loads through the standard transformers
audio-classification interface with no vendored architecture code. Heavy
imports are deferred until first model use; loading is thread-safe (the
`_load_once` idiom -- transformers weight loading is not thread-safe, and
concurrent loads return CORRUPTED WEIGHTS rather than an error; see the
comment in app/services/model_loader_service.py and commit 0c21127).

Which class index means "spoof" is read from the checkpoint's own
`config.json` at load time and never hardcoded -- getting it backwards
produces a detector that is confidently inverted with no error anywhere.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

TARGET_SAMPLE_RATE = 16_000

# Defensive cap. ASVspoof LA clips are a few seconds; this only bites on a
# long clip and is reported back so a truncated score is never silent.
MAX_ANALYSIS_SECONDS = 30.0

# No EER calibration exists yet -- Feature 1 (score distribution / DET curve)
# is what will produce one. Until then this is the naive midpoint and every
# response says so. Bump the version when a calibrated threshold replaces it:
# it is part of the cache key, so old results are invalidated.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_VERSION = "deepfake-threshold-v0-uncalibrated"


@dataclass(frozen=True, slots=True)
class DeepfakeModelSpec:
    key: str
    label: str
    model_id: str
    revision: str
    sampling_rate: int
    threshold: float
    threshold_calibrated: bool
    recommended: bool


MODEL_SPECS: dict[str, DeepfakeModelSpec] = {
    # Deliberately NOT "wav2vec2-xlsr-deepfake": the shared saliency service
    # dispatches on `"wav2vec" in model.lower()`
    # (app/services/saliency_service.py::detect_model_type), so an id carrying
    # that substring would silently route to the EMOTION model's saliency and
    # return emotion attributions labelled as deepfake attributions.
    "xlsr-deepfake": DeepfakeModelSpec(
        key="xlsr-deepfake",
        label="wav2vec2 XLS-R (Model A)",
        model_id="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        revision="main",
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=DEFAULT_THRESHOLD,
        threshold_calibrated=False,
        recommended=True,
    ),
}


class UnsupportedDeepfakeModel(ValueError):
    """Raised when the API receives a model key outside the task registry."""


class DeepfakeModelUnavailable(RuntimeError):
    """Raised when the checkpoint or its dependencies cannot load."""


class DeepfakeAdapter(Protocol):
    def score(self, audio_path: str | Path) -> dict: ...


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_SPOOF_TOKENS = {"spoof", "spoofed", "fake", "deepfake", "synthetic", "generated"}
_BONAFIDE_TOKENS = {"bonafide", "real", "genuine", "human", "authentic", "original"}


def _tokenize_label(label: str) -> set[str]:
    normalized = str(label).strip().lower().replace("-", " ").replace("_", " ")
    normalized = normalized.replace("bona fide", "bonafide")
    return set(normalized.split())


def resolve_spoof_index(id2label: dict) -> int:
    """Return the class index meaning "spoof", read from the checkpoint config.

    Raises `DeepfakeModelUnavailable` naming the observed mapping rather than
    guessing: a wrong guess inverts every score the task will ever produce,
    and nothing downstream can detect it.
    """

    normalized = {int(index): label for index, label in id2label.items()}
    if len(normalized) != 2:
        raise DeepfakeModelUnavailable(
            "Expected a binary bona fide/spoof classifier, but the checkpoint "
            f"declares {len(normalized)} classes: {normalized}."
        )

    spoof_indices = {
        index
        for index, label in normalized.items()
        if _tokenize_label(label) & _SPOOF_TOKENS
    }
    bonafide_indices = {
        index
        for index, label in normalized.items()
        if _tokenize_label(label) & _BONAFIDE_TOKENS
    }

    if len(spoof_indices) == 1 and not (spoof_indices & bonafide_indices):
        return spoof_indices.pop()
    if len(bonafide_indices) == 1 and not spoof_indices:
        (bonafide_index,) = bonafide_indices
        return next(index for index in normalized if index != bonafide_index)

    raise DeepfakeModelUnavailable(
        "Could not tell which class means 'spoof' from the checkpoint's own "
        f"labels: {normalized}. Resolve the mapping from the model card and "
        "set it explicitly before trusting any score."
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def _load_waveform(audio_path: str | Path):
    """Load mono float32 waveform at 16 kHz. Returns (tensor[1, n], sr).

    Decoding goes through soundfile (libsndfile), NOT `torchaudio.load`:
    torchaudio >= 2.9 delegates loading to TorchCodec, which is not installed
    here and raises ImportError. libsndfile reads both the WAV uploads and the
    FLAC files ASVspoof ships. Resampling still uses torchaudio.functional,
    which is a pure tensor op and needs no codec.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError(f"Could not decode audio: {audio_path} ({error})") from error

    if samples.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    # soundfile returns (frames, channels); torch wants (channels, frames).
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
    waveform = waveform.to(dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio contains invalid values: {audio_path}")
    return waveform, TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _HFAudioClassifierAdapter:
    """Wraps a standard transformers audio-classification checkpoint."""

    def __init__(self, spec: DeepfakeModelSpec) -> None:
        self.spec = spec
        try:
            import torch
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as error:
            raise DeepfakeModelUnavailable(
                "Deepfake detection requires 'transformers' and 'torch'. "
                "Install the backend requirements and restart the API."
            ) from error

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                spec.model_id, revision=spec.revision
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                spec.model_id, revision=spec.revision
            )
        except Exception as error:
            raise DeepfakeModelUnavailable(
                f"Could not load {spec.model_id}: {error}"
            ) from error

        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }
        self.spoof_index = resolve_spoof_index(self.id2label)
        self.bonafide_index = next(
            index for index in self.id2label if index != self.spoof_index
        )

    def score(self, audio_path: str | Path) -> dict:
        import torch

        waveform, sample_rate = _load_waveform(audio_path)
        duration = waveform.shape[1] / sample_rate

        max_samples = int(MAX_ANALYSIS_SECONDS * sample_rate)
        truncated = waveform.shape[1] > max_samples
        if truncated:
            waveform = waveform[:, :max_samples]

        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        return {
            "spoof_probability": round(float(probabilities[self.spoof_index]), 6),
            "bonafide_probability": round(
                float(probabilities[self.bonafide_index]), 6
            ),
            "logits": [round(v, 6) for v in logits.squeeze(0).cpu().tolist()],
            "id2label": self.id2label,
            "spoof_index": self.spoof_index,
            "duration": round(duration, 2),
            "analysed_seconds": round(min(duration, MAX_ANALYSIS_SECONDS), 2),
            "truncated": truncated,
        }


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, DeepfakeAdapter] = {}
_LOAD_LOCK = threading.Lock()


def list_models() -> list[dict[str, object]]:
    return [asdict(spec) for spec in MODEL_SPECS.values()]


def get_model_spec(model_key: str) -> DeepfakeModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        valid = ", ".join(MODEL_SPECS)
        raise UnsupportedDeepfakeModel(
            f"Unsupported deepfake model '{model_key}'. Valid models: {valid}."
        ) from error


def get_model(model_key: str) -> DeepfakeAdapter:
    """Load a selected model once; concurrent first requests cannot race."""
    spec = get_model_spec(model_key)
    with _LOAD_LOCK:
        if model_key not in _MODEL_CACHE:
            _MODEL_CACHE[model_key] = _HFAudioClassifierAdapter(spec)
        return _MODEL_CACHE[model_key]


def run_detection(model_key: str, audio_path: str | Path) -> dict:
    """Score one clip and apply the task's operating threshold."""
    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    result = adapter.score(audio_path)

    spoof_probability = result["spoof_probability"]
    return {
        "model": spec.key,
        "model_label": spec.label,
        "model_id": spec.model_id,
        "decision": "spoof" if spoof_probability >= spec.threshold else "bonafide",
        "threshold": spec.threshold,
        "threshold_calibrated": spec.threshold_calibrated,
        "threshold_version": THRESHOLD_VERSION,
        **result,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
