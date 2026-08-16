"""Single-clip perturbation comparison for Speaker Verification.

Applies exactly one user-selected perturbation to exactly one selected
recording, then compares that recording's own original and perturbed
embeddings directly -- no enrollment centroid, no separate probe/reference
clips. See `perturb_and_compare`.

Two correctness properties this module exists to guarantee:

* `apply_pitch_shift`/`apply_time_stretch` in `pertubation_service` silently
  fall back to the original, unmodified waveform on internal error, timeout,
  or a below-threshold parameter, with no flag anywhere signaling this. A
  naive caller could report a falsely "robust" result. `_waveform_changed`
  and `PerturbationNotApplied` exist specifically to catch that.
* `add_gaussian_noise` is the only stochastic transform. Its `seed` is
  request-scoped (a local `torch.Generator`, never global RNG state), so
  identical audio + params + seed always produces an identical result, and
  concurrent requests can never interfere with each other's noise draws.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.services import pertubation_service

from . import service

DEFAULT_NOISE_SEED = 0


class PerturbationNotApplied(ValueError):
    """The transform ran but produced audio identical to the input -- almost
    always pertubation_service's own silent-fallback path (timeout, internal
    exception, or a below-threshold effective parameter)."""


class PerturbationOutputInvalid(ValueError):
    """The transform produced a non-tensor, empty, non-numeric, or
    non-finite waveform."""


class _StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _reject_bool_and_nonfinite(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid numeric parameter")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("parameter must be finite (no NaN/Infinity)")
        return value


class NoiseParams(_StrictParams):
    noise_level: float = Field(gt=0, le=0.5)
    seed: int = Field(default=DEFAULT_NOISE_SEED, ge=0, le=2**32 - 1)


class TimeMaskingParams(_StrictParams):
    mask_start_percent: float = Field(ge=0, le=100)
    mask_end_percent: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _check_order(self) -> "TimeMaskingParams":
        if self.mask_start_percent >= self.mask_end_percent:
            raise ValueError("mask_start_percent must be less than mask_end_percent")
        return self


class FrequencyMaskingParams(_StrictParams):
    mask_freq_start: float = Field(ge=0)
    mask_freq_end: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> "FrequencyMaskingParams":
        if self.mask_freq_start >= self.mask_freq_end:
            raise ValueError("mask_freq_start must be less than mask_freq_end")
        return self


class PitchShiftParams(_StrictParams):
    pitch_shift_semitones: float = Field(ge=-6, le=6)

    @field_validator("pitch_shift_semitones")
    @classmethod
    def _reject_near_zero(cls, value: float) -> float:
        if abs(value) < 0.1:
            raise ValueError("pitch_shift_semitones must have magnitude >= 0.1")
        return value


class TimeStretchParams(_StrictParams):
    stretch_factor: float = Field(gt=0, ge=0.5, le=2.0)

    @field_validator("stretch_factor")
    @classmethod
    def _reject_near_one(cls, value: float) -> float:
        if abs(value - 1.0) < 0.01:
            raise ValueError("stretch_factor must differ from 1.0 by at least 0.01")
        return value


PARAM_MODELS: dict[str, type[_StrictParams]] = {
    "noise": NoiseParams,
    "time_masking": TimeMaskingParams,
    "frequency_masking": FrequencyMaskingParams,
    "pitch_shift": PitchShiftParams,
    "time_stretch": TimeStretchParams,
}


def validate_perturbation_params(
    perturbation_type: str, raw_params: dict[str, Any], *, sample_rate: int | None = None
) -> dict[str, Any]:
    """Exactly the keys applicable to `perturbation_type`; nothing missing,
    nothing extra, no bools-as-numbers, no NaN/Infinity. Returns the
    normalized params actually used (including the noise seed)."""

    model_cls = PARAM_MODELS.get(perturbation_type)
    if model_cls is None:
        valid = ", ".join(PARAM_MODELS)
        raise ValueError(f"Unsupported perturbation type '{perturbation_type}'. Valid types: {valid}.")
    try:
        parsed = model_cls.model_validate(raw_params)
    except ValidationError as error:
        raise ValueError(str(error)) from error

    if isinstance(parsed, FrequencyMaskingParams) and sample_rate is not None:
        nyquist = sample_rate / 2
        if parsed.mask_freq_end > nyquist:
            raise ValueError(f"mask_freq_end must be <= Nyquist frequency ({nyquist} Hz).")

    return parsed.model_dump()


def _apply_single_perturbation(
    waveform: torch.Tensor, sample_rate: int, perturbation_type: str, params: dict[str, Any]
) -> torch.Tensor:
    """Dispatch through the `pertubation_service` module (not bound function
    references), so test-time monkeypatching of its silent-fallback path is
    always observed."""

    if perturbation_type == "noise":
        return pertubation_service.add_gaussian_noise(waveform, params["noise_level"], seed=params["seed"])
    if perturbation_type == "time_masking":
        return pertubation_service.apply_time_masking(
            waveform, params["mask_start_percent"], params["mask_end_percent"]
        )
    if perturbation_type == "frequency_masking":
        return pertubation_service.apply_frequency_masking(
            waveform, sample_rate, params["mask_freq_start"], params["mask_freq_end"]
        )
    if perturbation_type == "pitch_shift":
        return pertubation_service.apply_pitch_shift(waveform, sample_rate, params["pitch_shift_semitones"])
    if perturbation_type == "time_stretch":
        return pertubation_service.apply_time_stretch(waveform, params["stretch_factor"])
    raise ValueError(f"Unsupported perturbation type '{perturbation_type}'.")


def _validate_output_waveform(tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise PerturbationOutputInvalid("Perturbation produced a non-tensor output.")
    if tensor.numel() == 0:
        raise PerturbationOutputInvalid("Perturbation produced an empty waveform.")
    if not torch.is_floating_point(tensor):
        raise PerturbationOutputInvalid("Perturbation produced a non-numeric waveform.")
    if not torch.isfinite(tensor).all():
        raise PerturbationOutputInvalid("Perturbation produced a non-finite waveform.")


def _waveform_changed(original: torch.Tensor, perturbed: torch.Tensor) -> bool:
    """Exact content comparison, not a tolerance-based one: a small but
    genuine perturbation (e.g. very low noise_level) must never be
    misclassified as a no-op because of a loose atol. Shape mismatch (e.g. a
    real time-stretch) always counts as changed. Dtype is never downcast for
    the comparison -- if the two tensors' dtypes differ, both are promoted
    (never truncated) to their common dtype before the exact torch.equal
    check, so precision can only increase, never be erased."""

    a = original.detach().cpu()
    b = perturbed.detach().cpu()
    if a.shape != b.shape:
        return True
    if a.dtype != b.dtype:
        common_dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=common_dtype)
        b = b.to(dtype=common_dtype)
    return not torch.equal(a, b)


def _load_mono_waveform(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError("Probe audio is empty.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def perturb_and_compare(
    model_key: str,
    source_path: str | Path,
    perturbation_type: str,
    perturbation_params: dict[str, Any],
    output_path: str | Path,
    *,
    cached_original_embedding: torch.Tensor | None = None,
) -> dict[str, object]:
    """Apply one perturbation to one recording and compare its own original
    and perturbed embeddings. Model loading and all embedding inference are
    deferred until the perturbation is confirmed valid and actually applied,
    and `output_path` is never written before that point either, so a bad
    request or a silent-no-op transform fails cheaply and leaves nothing on
    disk. The perturbed waveform is written directly to the caller-supplied
    `output_path` (the session asset's own temp path), not a scratch
    directory, and one loaded adapter is reused for both embeddings."""

    spec = service.get_model_spec(model_key)

    waveform, sample_rate = _load_mono_waveform(source_path)

    normalized_params = validate_perturbation_params(
        perturbation_type, perturbation_params, sample_rate=sample_rate
    )
    perturbed_waveform = _apply_single_perturbation(waveform, sample_rate, perturbation_type, normalized_params)
    _validate_output_waveform(perturbed_waveform)
    if not _waveform_changed(waveform, perturbed_waveform):
        raise PerturbationNotApplied(
            f"The requested '{perturbation_type}' perturbation did not change the audio. "
            "This usually means the underlying transform silently failed (e.g. a pitch-shift "
            "timeout) rather than that the model is robust to it."
        )

    adapter = service.get_model(model_key)
    original_embedding = cached_original_embedding
    if original_embedding is None:
        original_embedding = adapter.extract_embedding(source_path)

    torchaudio.save(str(output_path), perturbed_waveform, sample_rate)
    perturbed_embedding = adapter.extract_embedding(output_path)

    similarity = service._cosine(original_embedding, perturbed_embedding)
    same_speaker = similarity >= spec.threshold

    return {
        "model": spec.key,
        "model_label": spec.label,
        "threshold": spec.threshold,
        "perturbation": {"type": perturbation_type, "params": normalized_params},
        "similarity": similarity,
        "same_speaker": same_speaker,
        "embedding_shift": 1.0 - similarity,
        "original_embedding": original_embedding.tolist(),
        "perturbed_embedding": perturbed_embedding.tolist(),
    }
