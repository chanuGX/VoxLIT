"""Focused tests for Speaker Verification perturbation-robustness analysis."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from app.core.settings import settings
from app.services import pertubation_service
from app.tasks.verification import dataset, robustness, service

ECAPA_SPEC = service.MODEL_SPECS["ecapa-tdnn"]
RESNET_SPEC = service.MODEL_SPECS["resnet34-lm"]


# --------------------------------------------------------------------------
# Audio fixtures
# --------------------------------------------------------------------------


def _write_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 220.0) -> Path:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return path


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def enrollment_and_probe(tmp_path):
    enrollment = [_write_wav(tmp_path / f"enrollment-{i}.wav", freq=200.0 + i * 15) for i in range(3)]
    probe = _write_wav(tmp_path / "probe.wav", freq=333.0)
    return enrollment, probe


FAKE_ROBUSTNESS_FILES = [f"speaker{i}_clip.wav" for i in range(5)]


def _build_fake_robustness_dataset(root: Path) -> Path:
    dataset_dir = root / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for index, filename in enumerate(FAKE_ROBUSTNESS_FILES):
        _write_wav(dataset_dir / filename, freq=180.0 + index * 20)
    return dataset_dir


@pytest.fixture
def fake_robustness_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    _build_fake_robustness_dataset(tmp_path)
    return [recording.recording_id for recording in dataset.list_recordings()]


# --------------------------------------------------------------------------
# Fake adapters
# --------------------------------------------------------------------------


class _HashDerivedAdapter:
    """Embedding is a deterministic function of the audio file's exact
    bytes: identical bytes (e.g. same noise seed) always yield identical
    embeddings, different bytes yield different embeddings, without
    pre-registering every possible perturbed-content variant."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def extract_embedding(self, audio_path):
        content = Path(audio_path).read_bytes()
        import hashlib

        digest = hashlib.sha256(content).digest()
        values = torch.tensor([digest[i % len(digest)] for i in range(self.dimension)], dtype=torch.float32)
        return torch.nn.functional.normalize(values, p=2, dim=0)


class _TwoValueAdapter:
    """Enrollment and the original probe get `same_embedding`; the
    perturbed probe (identified by the `probe-perturbed.wav` filename
    `robustness_analysis` always writes) gets `perturbed_embedding`."""

    def __init__(self, same_embedding: torch.Tensor, perturbed_embedding: torch.Tensor) -> None:
        self.same_embedding = same_embedding
        self.perturbed_embedding = perturbed_embedding

    def extract_embedding(self, audio_path):
        if "probe-perturbed.wav" in str(audio_path):
            return self.perturbed_embedding
        return self.same_embedding


class _FailingOnPerturbedAdapter:
    """Raises only when asked for the perturbed-probe embedding, to exercise
    temp-file cleanup on an inference error."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def extract_embedding(self, audio_path):
        if "probe-perturbed.wav" in str(audio_path):
            raise RuntimeError("boom")
        values = torch.zeros(self.dimension)
        values[0] = 1.0
        return values


class _ModelLoadForbidden:
    def __call__(self, _model_key):
        raise AssertionError("model must not be loaded for an invalid/no-op perturbation")


class _RecordingTemporaryDirectory:
    created: list[str] = []

    def __init__(self, *args, **kwargs):
        import tempfile as _tempfile_module

        self._impl = _tempfile_module.TemporaryDirectory(*args, **kwargs)

    def __enter__(self):
        name = self._impl.__enter__()
        _RecordingTemporaryDirectory.created.append(name)
        return name

    def __exit__(self, *exc_info):
        return self._impl.__exit__(*exc_info)


class _TimingOutProcess:
    """Stands in for `pertubation_service._MP_CONTEXT.Process`: alive until
    terminate()/kill() is called, mirroring a genuine pitch-shift timeout
    without any real waiting."""

    def __init__(self, target=None, args=(), daemon=None) -> None:
        self.args = args
        self._alive = True

    def start(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        pass

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# _waveform_changed -- exact comparison, no precision loss
# --------------------------------------------------------------------------


def test_waveform_changed_true_for_shape_mismatch():
    original = torch.zeros(1, 100)
    perturbed = torch.zeros(1, 150)
    assert robustness._waveform_changed(original, perturbed) is True


def test_waveform_changed_false_for_identical_tensor():
    original = torch.rand(1, 100)
    assert robustness._waveform_changed(original, original.clone()) is False


def test_waveform_changed_detects_tiny_float64_difference():
    """A difference far below float32 precision must still be detected --
    proves the comparison never downcasts before checking equality."""

    original = torch.full((1, 4), 1.0, dtype=torch.float64)
    perturbed = original.clone()
    perturbed[0, 0] = 1.0 + 1e-9
    assert robustness._waveform_changed(original, perturbed) is True


# --------------------------------------------------------------------------
# _validate_output_waveform
# --------------------------------------------------------------------------


def test_validate_output_waveform_rejects_non_tensor():
    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness._validate_output_waveform([1.0, 2.0, 3.0])


def test_validate_output_waveform_rejects_empty_tensor():
    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness._validate_output_waveform(torch.zeros(1, 0))


def test_validate_output_waveform_rejects_non_finite():
    tensor = torch.zeros(1, 4)
    tensor[0, 0] = float("nan")
    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness._validate_output_waveform(tensor)


def test_validate_output_waveform_accepts_valid_tensor():
    robustness._validate_output_waveform(torch.rand(1, 100))


# --------------------------------------------------------------------------
# Strict parameter validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_type,params",
    [
        ("noise", {"seed": 0}),
        ("time_masking", {"mask_start_percent": 10}),
        ("frequency_masking", {"mask_freq_start": 100}),
        ("pitch_shift", {}),
        ("time_stretch", {}),
    ],
)
def test_missing_required_key_rejected(perturbation_type, params):
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params(perturbation_type, params)


def test_extra_key_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("noise", {"noise_level": 0.05, "pitch_shift_semitones": 2.0})


def test_boolean_noise_level_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("noise", {"noise_level": True})


def test_boolean_seed_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("noise", {"noise_level": 0.05, "seed": True})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_value_rejected(bad_value):
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("noise", {"noise_level": bad_value})


def test_time_masking_inverted_range_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params(
            "time_masking", {"mask_start_percent": 50, "mask_end_percent": 50}
        )


def test_frequency_masking_inverted_range_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params(
            "frequency_masking", {"mask_freq_start": 2000, "mask_freq_end": 2000}
        )


def test_frequency_masking_above_nyquist_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params(
            "frequency_masking",
            {"mask_freq_start": 1000, "mask_freq_end": 9000},
            sample_rate=16000,
        )


def test_pitch_shift_near_zero_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("pitch_shift", {"pitch_shift_semitones": 0.05})


@pytest.mark.parametrize("value", [-7.0, 7.0])
def test_pitch_shift_out_of_bounds_rejected(value):
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("pitch_shift", {"pitch_shift_semitones": value})


def test_time_stretch_near_one_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("time_stretch", {"stretch_factor": 1.005})


@pytest.mark.parametrize("value", [0.4, 2.5])
def test_time_stretch_out_of_bounds_rejected(value):
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("time_stretch", {"stretch_factor": value})


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_noise_seed_out_of_range_rejected(seed):
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("noise", {"noise_level": 0.05, "seed": seed})


def test_unsupported_perturbation_type_rejected():
    with pytest.raises(ValueError):
        robustness.validate_perturbation_params("unknown_type", {})


def test_valid_noise_params_normalized_with_default_seed():
    result = robustness.validate_perturbation_params("noise", {"noise_level": 0.02})
    assert result == {"noise_level": 0.02, "seed": robustness.DEFAULT_NOISE_SEED}


# --------------------------------------------------------------------------
# Reproducibility (seeded noise)
# --------------------------------------------------------------------------


def test_add_gaussian_noise_same_seed_produces_identical_tensor():
    waveform = torch.zeros(1, 4000)
    a = pertubation_service.add_gaussian_noise(waveform, 0.05, seed=42)
    b = pertubation_service.add_gaussian_noise(waveform, 0.05, seed=42)
    assert torch.equal(a, b)


def test_add_gaussian_noise_different_seed_produces_different_tensor():
    waveform = torch.zeros(1, 4000)
    a = pertubation_service.add_gaussian_noise(waveform, 0.05, seed=1)
    b = pertubation_service.add_gaussian_noise(waveform, 0.05, seed=2)
    assert not torch.equal(a, b)


def test_add_gaussian_noise_unseeded_call_path_still_works():
    """Regression guard: the original, unseeded behaviour must be unchanged."""
    waveform = torch.zeros(1, 100)
    result = pertubation_service.add_gaussian_noise(waveform, 0.05)
    assert result.shape == waveform.shape


def test_noise_perturbation_same_seed_is_reproducible(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    first = robustness.robustness_analysis(
        "ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05, "seed": 7}
    )
    second = robustness.robustness_analysis(
        "ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05, "seed": 7}
    )

    assert first["perturbed_similarity"] == pytest.approx(second["perturbed_similarity"])
    assert first["perturbation"]["params"]["seed"] == 7


def test_noise_perturbation_different_seed_changes_result(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    first = robustness.robustness_analysis(
        "ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05, "seed": 1}
    )
    second = robustness.robustness_analysis(
        "ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05, "seed": 2}
    )

    assert first["perturbed_similarity"] != second["perturbed_similarity"]


# --------------------------------------------------------------------------
# Both production thresholds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_key,spec", [("ecapa-tdnn", ECAPA_SPEC), ("resnet34-lm", RESNET_SPEC)])
def test_response_reports_model_specific_threshold(monkeypatch, enrollment_and_probe, model_key, spec):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: spec)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(spec.embedding_dimension))

    result = robustness.robustness_analysis(model_key, enrollment, probe, "noise", {"noise_level": 0.05})

    assert result["threshold"] == spec.threshold
    assert result["model"] == spec.key


# --------------------------------------------------------------------------
# Signed delta / decision flip
# --------------------------------------------------------------------------


def test_signed_delta_and_decision_flip_on_threshold_crossing(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    dim = ECAPA_SPEC.embedding_dimension
    same = torch.nn.functional.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    different = torch.nn.functional.normalize(torch.tensor([0.0, 1.0] + [0.0] * (dim - 2)), p=2, dim=0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _TwoValueAdapter(same, different))

    result = robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})

    assert result["original_similarity"] == pytest.approx(1.0)
    assert result["perturbed_similarity"] == pytest.approx(0.0, abs=1e-6)
    assert result["similarity_delta"] == pytest.approx(
        result["perturbed_similarity"] - result["original_similarity"]
    )
    assert result["similarity_change_abs"] == pytest.approx(abs(result["similarity_delta"]))
    assert result["original_same_speaker"] is True
    assert result["perturbed_same_speaker"] is False
    assert result["decision_flipped"] is True


def test_decision_not_flipped_when_still_above_threshold(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    dim = ECAPA_SPEC.embedding_dimension
    same = torch.nn.functional.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    slightly_off = torch.nn.functional.normalize(torch.tensor([0.99, 0.05] + [0.0] * (dim - 2)), p=2, dim=0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _TwoValueAdapter(same, slightly_off))

    result = robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})

    assert result["perturbed_similarity"] >= ECAPA_SPEC.threshold
    assert result["decision_flipped"] is False


# --------------------------------------------------------------------------
# No-op rejection reaches through the module import; deferred model loading
# --------------------------------------------------------------------------


def test_pitch_shift_noop_fallback_is_rejected(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", lambda waveform, *a, **k: waveform)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.robustness_analysis(
            "ecapa-tdnn", enrollment, probe, "pitch_shift", {"pitch_shift_semitones": 2.0}
        )


def test_time_stretch_noop_fallback_is_rejected(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_time_stretch", lambda waveform, *a, **k: waveform)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.robustness_analysis(
            "ecapa-tdnn", enrollment, probe, "time_stretch", {"stretch_factor": 1.5}
        )


def test_noop_perturbation_never_loads_model(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", lambda waveform, *a, **k: waveform)
    monkeypatch.setattr(service, "get_model", _ModelLoadForbidden())

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.robustness_analysis(
            "ecapa-tdnn", enrollment, probe, "pitch_shift", {"pitch_shift_semitones": 2.0}
        )


def test_invalid_params_never_load_model(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", _ModelLoadForbidden())

    with pytest.raises(ValueError):
        robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": -1})


# --------------------------------------------------------------------------
# Non-finite/invalid output rejection
# --------------------------------------------------------------------------


def test_non_finite_output_rejected(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)

    def _nan_noise(waveform, *_args, **_kwargs):
        result = waveform.clone()
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(pertubation_service, "add_gaussian_noise", _nan_noise)

    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})


def test_empty_output_rejected(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "add_gaussian_noise", lambda waveform, *a, **k: torch.zeros(1, 0))

    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})


# --------------------------------------------------------------------------
# Real (non-monkeypatched) transforms
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_type,params",
    [
        ("noise", {"noise_level": 0.05}),
        ("time_masking", {"mask_start_percent": 20, "mask_end_percent": 40}),
        ("frequency_masking", {"mask_freq_start": 500, "mask_freq_end": 2000}),
    ],
)
def test_real_transforms_apply_successfully(monkeypatch, enrollment_and_probe, perturbation_type, params):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, perturbation_type, params)

    assert result["perturbation_applied"] is True


# --------------------------------------------------------------------------
# 2D/mono and sample-rate handling
# --------------------------------------------------------------------------


def test_stereo_probe_is_downmixed_before_perturbation(monkeypatch, tmp_path):
    enrollment = [_write_wav(tmp_path / f"enrollment-{i}.wav", freq=200.0 + i * 15) for i in range(3)]
    stereo_path = tmp_path / "probe-stereo.wav"
    t = np.linspace(0, 1.0, 16000, False)
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 240.0 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(stereo_path, stereo, 16000)

    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.robustness_analysis(
        "ecapa-tdnn", enrollment, stereo_path, "noise", {"noise_level": 0.05}
    )

    assert result["perturbation_applied"] is True


def test_non_16khz_probe_still_produces_valid_result(monkeypatch, tmp_path):
    enrollment = [_write_wav(tmp_path / f"enrollment-{i}.wav", freq=200.0 + i * 15) for i in range(3)]
    probe = _write_wav(tmp_path / "probe-8k.wav", sample_rate=8000, freq=300.0)

    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})

    assert result["perturbation_applied"] is True


# --------------------------------------------------------------------------
# Temp-file cleanup
# --------------------------------------------------------------------------


def test_perturbed_audio_temp_dir_cleaned_up_on_success(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    _RecordingTemporaryDirectory.created.clear()
    monkeypatch.setattr(robustness, "TemporaryDirectory", _RecordingTemporaryDirectory)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})

    assert _RecordingTemporaryDirectory.created
    assert not Path(_RecordingTemporaryDirectory.created[-1]).exists()


def test_perturbed_audio_temp_dir_cleaned_up_on_inference_error(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    _RecordingTemporaryDirectory.created.clear()
    monkeypatch.setattr(robustness, "TemporaryDirectory", _RecordingTemporaryDirectory)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _FailingOnPerturbedAdapter(ECAPA_SPEC.embedding_dimension))

    with pytest.raises(RuntimeError):
        robustness.robustness_analysis("ecapa-tdnn", enrollment, probe, "noise", {"noise_level": 0.05})

    assert _RecordingTemporaryDirectory.created
    assert not Path(_RecordingTemporaryDirectory.created[-1]).exists()


# --------------------------------------------------------------------------
# Perturbation exceptions/timeouts through robustness_analysis
# --------------------------------------------------------------------------


def test_pitch_shift_timeout_is_rejected_not_hung(monkeypatch, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service._MP_CONTEXT, "Process", _TimingOutProcess)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.robustness_analysis(
            "ecapa-tdnn", enrollment, probe, "pitch_shift", {"pitch_shift_semitones": 2.0}
        )


# --------------------------------------------------------------------------
# Router: dataset variant
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_rejects_duplicate_enrollment_ids(client, fake_robustness_dataset):
    ids = fake_robustness_dataset[:3]
    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": [ids[0], ids[0], ids[1]],
        "probe_recording_id": ids[2],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 422
    assert "Duplicate" in response.json()["detail"]


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_rejects_probe_in_enrollment(client, fake_robustness_dataset):
    ids = fake_robustness_dataset[:3]
    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": ids,
        "probe_recording_id": ids[0],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_unsupported_model_is_400(client, fake_robustness_dataset):
    ids = fake_robustness_dataset[:4]
    payload = {
        "model": "not-a-real-model",
        "enrollment_recording_ids": ids[:3],
        "probe_recording_id": ids[3],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_unknown_recording_id_is_404(client, fake_robustness_dataset):
    ids = fake_robustness_dataset[:3]
    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": ids,
        "probe_recording_id": "rec_0000000000000000",
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_missing_dataset_is_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path / "does-not-exist")
    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": ["rec_a", "rec_b", "rec_c"],
        "probe_recording_id": "rec_d",
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_rejects_traversal_id(client, fake_robustness_dataset):
    """A path-traversal-shaped id never touches the filesystem -- it just
    fails the opaque-id hash comparison like any other unknown id, and 404s.
    `RecordingNotFound` echoes back the client's own supplied id (not real
    dataset content), matching `resolve_recording_path`'s existing,
    unchanged behavior used by `/batch/dataset`."""

    ids = fake_robustness_dataset[:3]
    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": ids,
        "probe_recording_id": "../../etc/passwd",
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)
    assert response.status_code == 404
    for real_filename in FAKE_ROBUSTNESS_FILES:
        assert real_filename not in response.text
    assert "vox_indian_demo_92" not in response.text


@pytest.mark.asyncio
async def test_robustness_dataset_endpoint_happy_path_no_leak(monkeypatch, client, fake_robustness_dataset):
    ids = fake_robustness_dataset[:4]
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    payload = {
        "model": "ecapa-tdnn",
        "enrollment_recording_ids": ids[:3],
        "probe_recording_id": ids[3],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05, "seed": 3}},
    }
    response = await client.post("/tasks/verification/robustness/dataset", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["perturbation_applied"] is True
    assert body["enrollment_recording_ids"] == ids[:3]
    assert body["probe_recording_id"] == ids[3]

    serialized = json.dumps(body)
    for filename in FAKE_ROBUSTNESS_FILES:
        assert filename not in serialized
    assert "vox_indian_demo_92" not in serialized


# --------------------------------------------------------------------------
# Router: upload variant
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_rejects_duplicate_enrollment_content(client):
    content = _wav_bytes(freq=220.0)
    files = [
        ("enrollment_files", ("a.wav", io.BytesIO(content), "audio/wav")),
        ("enrollment_files", ("b_copy.wav", io.BytesIO(content), "audio/wav")),
        ("enrollment_files", ("c.wav", io.BytesIO(_wav_bytes(freq=300.0)), "audio/wav")),
        ("probe_file", ("probe.wav", io.BytesIO(_wav_bytes(freq=250.0)), "audio/wav")),
    ]
    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "ecapa-tdnn", "perturbation_type": "noise", "noise_level": "0.05"},
        files=files,
    )
    assert response.status_code == 422
    assert "identical audio content" in response.json()["detail"]


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_rejects_probe_matching_enrollment_content(client):
    content = _wav_bytes(freq=220.0)
    files = [
        ("enrollment_files", ("a.wav", io.BytesIO(content), "audio/wav")),
        ("enrollment_files", ("b.wav", io.BytesIO(_wav_bytes(freq=260.0)), "audio/wav")),
        ("enrollment_files", ("c.wav", io.BytesIO(_wav_bytes(freq=300.0)), "audio/wav")),
        ("probe_file", ("probe_copy.wav", io.BytesIO(content), "audio/wav")),
    ]
    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "ecapa-tdnn", "perturbation_type": "noise", "noise_level": "0.05"},
        files=files,
    )
    assert response.status_code == 422
    assert "same audio content" in response.json()["detail"]


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_unsupported_model_is_400(client):
    files = [
        ("enrollment_files", ("a.wav", io.BytesIO(_wav_bytes(freq=200.0)), "audio/wav")),
        ("enrollment_files", ("b.wav", io.BytesIO(_wav_bytes(freq=210.0)), "audio/wav")),
        ("enrollment_files", ("c.wav", io.BytesIO(_wav_bytes(freq=220.0)), "audio/wav")),
        ("probe_file", ("probe.wav", io.BytesIO(_wav_bytes(freq=230.0)), "audio/wav")),
    ]
    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "not-a-real-model", "perturbation_type": "noise", "noise_level": "0.05"},
        files=files,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_requires_three_enrollment_files(client):
    files = [
        ("enrollment_files", ("a.wav", io.BytesIO(_wav_bytes(freq=200.0)), "audio/wav")),
        ("probe_file", ("probe.wav", io.BytesIO(_wav_bytes(freq=230.0)), "audio/wav")),
    ]
    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "ecapa-tdnn", "perturbation_type": "noise", "noise_level": "0.05"},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_happy_path_no_filename_leak(monkeypatch, client):
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))
    filenames = ["alice.wav", "bob.wav", "carol.wav", "dave-probe.wav"]
    files = [
        ("enrollment_files", (filenames[0], io.BytesIO(_wav_bytes(freq=200.0)), "audio/wav")),
        ("enrollment_files", (filenames[1], io.BytesIO(_wav_bytes(freq=210.0)), "audio/wav")),
        ("enrollment_files", (filenames[2], io.BytesIO(_wav_bytes(freq=220.0)), "audio/wav")),
        ("probe_file", (filenames[3], io.BytesIO(_wav_bytes(freq=230.0)), "audio/wav")),
    ]
    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "ecapa-tdnn", "perturbation_type": "noise", "noise_level": "0.05", "seed": "5"},
        files=files,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["probe_label"] == "probe"
    assert body["perturbation"]["params"]["seed"] == 5
    serialized = json.dumps(body)
    for filename in filenames:
        assert filename not in serialized


@pytest.mark.asyncio
async def test_robustness_upload_endpoint_pitch_shift_timeout_is_422(monkeypatch, client, enrollment_and_probe):
    enrollment, probe = enrollment_and_probe
    monkeypatch.setattr(pertubation_service._MP_CONTEXT, "Process", _TimingOutProcess)

    files = [("enrollment_files", (p.name, io.BytesIO(p.read_bytes()), "audio/wav")) for p in enrollment]
    files.append(("probe_file", (probe.name, io.BytesIO(probe.read_bytes()), "audio/wav")))

    response = await client.post(
        "/tasks/verification/robustness/upload",
        data={"model": "ecapa-tdnn", "perturbation_type": "pitch_shift", "pitch_shift_semitones": "2.0"},
        files=files,
    )

    assert response.status_code == 422
