"""Focused tests for Speaker Verification single-clip perturbation comparison."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from httpx import AsyncClient

from app.core.settings import settings
from app.main import app
from app.services import pertubation_service
from app.tasks.verification import cache, dataset, robustness, service

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
def source_and_output(tmp_path):
    source = _write_wav(tmp_path / "source.wav", freq=333.0)
    output = tmp_path / "output.wav"
    return source, output


FAKE_FILES = [f"speaker{i}_clip.wav" for i in range(5)]


def _build_fake_dataset(root: Path) -> Path:
    dataset_dir = root / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for index, filename in enumerate(FAKE_FILES):
        _write_wav(dataset_dir / filename, freq=180.0 + index * 20)
    return dataset_dir


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    _build_fake_dataset(tmp_path)
    return [recording.recording_id for recording in dataset.list_recordings()]


def _assert_no_ground_truth_leak(payload) -> None:
    serialized = json.dumps(payload)
    for filename in FAKE_FILES:
        assert filename not in serialized
    assert "vox_indian_demo_92" not in serialized


# --------------------------------------------------------------------------
# Fake adapters
# --------------------------------------------------------------------------


class _FakeAdapter:
    """Embedding keyed by the exact path string `perturb_and_compare` used
    to request it -- the same pattern `service.verify_speaker`'s tests use."""

    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def extract_embedding(self, audio_path):
        return self.embeddings[str(audio_path)]


class _HashDerivedAdapter:
    """Embedding is a deterministic function of the audio file's exact
    bytes: identical bytes (e.g. same noise seed) always yield identical
    embeddings, different bytes yield different embeddings."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def extract_embedding(self, audio_path):
        content = Path(audio_path).read_bytes()
        import hashlib

        digest = hashlib.sha256(content).digest()
        values = torch.tensor([digest[i % len(digest)] for i in range(self.dimension)], dtype=torch.float32)
        return F.normalize(values, p=2, dim=0)


class _ModelLoadForbidden:
    def __call__(self, _model_key):
        raise AssertionError("model must not be loaded for an invalid/no-op perturbation")


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
# Reproducibility (seeded noise) -- direct pertubation_service calls
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


def test_noise_perturbation_same_seed_is_reproducible(monkeypatch, tmp_path):
    source = _write_wav(tmp_path / "source.wav", freq=333.0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    first = robustness.perturb_and_compare(
        "ecapa-tdnn", source, "noise", {"noise_level": 0.05, "seed": 7}, tmp_path / "out1.wav"
    )
    second = robustness.perturb_and_compare(
        "ecapa-tdnn", source, "noise", {"noise_level": 0.05, "seed": 7}, tmp_path / "out2.wav"
    )

    assert first["similarity"] == pytest.approx(second["similarity"])
    assert first["perturbation"]["params"]["seed"] == 7


def test_noise_perturbation_different_seed_changes_result(monkeypatch, tmp_path):
    source = _write_wav(tmp_path / "source.wav", freq=333.0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    first = robustness.perturb_and_compare(
        "ecapa-tdnn", source, "noise", {"noise_level": 0.05, "seed": 1}, tmp_path / "out1.wav"
    )
    second = robustness.perturb_and_compare(
        "ecapa-tdnn", source, "noise", {"noise_level": 0.05, "seed": 2}, tmp_path / "out2.wav"
    )

    assert first["similarity"] != second["similarity"]


# --------------------------------------------------------------------------
# Both production thresholds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_key,spec", [("ecapa-tdnn", ECAPA_SPEC), ("resnet34-lm", RESNET_SPEC)])
def test_response_reports_model_specific_threshold(monkeypatch, source_and_output, model_key, spec):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: spec)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(spec.embedding_dimension))

    result = robustness.perturb_and_compare(model_key, source, "noise", {"noise_level": 0.05}, output)

    assert result["threshold"] == spec.threshold
    assert result["model"] == spec.key


# --------------------------------------------------------------------------
# similarity / embedding_shift / same_speaker
# --------------------------------------------------------------------------


def test_same_speaker_true_and_zero_shift_for_identical_embeddings(monkeypatch, source_and_output):
    source, output = source_and_output
    dim = ECAPA_SPEC.embedding_dimension
    same = F.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(
        service, "get_model", lambda _: _FakeAdapter({str(source): same, str(output): same})
    )

    result = robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)

    assert result["similarity"] == pytest.approx(1.0)
    assert result["same_speaker"] is True
    assert result["embedding_shift"] == pytest.approx(0.0, abs=1e-6)


def test_same_speaker_false_and_full_shift_for_orthogonal_embeddings(monkeypatch, source_and_output):
    source, output = source_and_output
    dim = ECAPA_SPEC.embedding_dimension
    original = F.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    perturbed = F.normalize(torch.tensor([0.0, 1.0] + [0.0] * (dim - 2)), p=2, dim=0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(
        service, "get_model", lambda _: _FakeAdapter({str(source): original, str(output): perturbed})
    )

    result = robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)

    assert result["similarity"] == pytest.approx(0.0, abs=1e-6)
    assert result["same_speaker"] is False
    assert result["embedding_shift"] == pytest.approx(1.0, abs=1e-6)
    assert result["embedding_shift"] == pytest.approx(1.0 - result["similarity"])


# --------------------------------------------------------------------------
# Adapter reuse / cached original embedding
# --------------------------------------------------------------------------


def test_adapter_loaded_exactly_once(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    calls = []

    def _spy_get_model(model_key):
        calls.append(model_key)
        return _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension)

    monkeypatch.setattr(service, "get_model", _spy_get_model)

    robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)

    assert calls == ["ecapa-tdnn"]


def test_cached_original_embedding_skips_source_extraction(monkeypatch, source_and_output):
    source, output = source_and_output
    dim = ECAPA_SPEC.embedding_dimension
    cached = F.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    perturbed = F.normalize(torch.tensor([0.0, 1.0] + [0.0] * (dim - 2)), p=2, dim=0)
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    # No entry for `source` -- would KeyError if the source were re-extracted.
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter({str(output): perturbed}))

    result = robustness.perturb_and_compare(
        "ecapa-tdnn",
        source,
        "noise",
        {"noise_level": 0.05},
        output,
        cached_original_embedding=cached,
    )

    assert result["similarity"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# No-op / invalid-parameter rejection -- deferred model loading, no output file
# --------------------------------------------------------------------------


def test_pitch_shift_noop_fallback_is_rejected(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", lambda waveform, *a, **k: waveform)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.perturb_and_compare(
            "ecapa-tdnn", source, "pitch_shift", {"pitch_shift_semitones": 2.0}, output
        )
    assert not output.exists()


def test_time_stretch_noop_fallback_is_rejected(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_time_stretch", lambda waveform, *a, **k: waveform)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.perturb_and_compare("ecapa-tdnn", source, "time_stretch", {"stretch_factor": 1.5}, output)
    assert not output.exists()


def test_noop_perturbation_never_loads_model(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", lambda waveform, *a, **k: waveform)
    monkeypatch.setattr(service, "get_model", _ModelLoadForbidden())

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.perturb_and_compare(
            "ecapa-tdnn", source, "pitch_shift", {"pitch_shift_semitones": 2.0}, output
        )
    assert not output.exists()


def test_invalid_params_never_load_model(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", _ModelLoadForbidden())

    with pytest.raises(ValueError):
        robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": -1}, output)
    assert not output.exists()


def test_pitch_shift_timeout_is_rejected_not_hung(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service._MP_CONTEXT, "Process", _TimingOutProcess)

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.perturb_and_compare(
            "ecapa-tdnn", source, "pitch_shift", {"pitch_shift_semitones": 2.0}, output
        )
    assert not output.exists()


# --------------------------------------------------------------------------
# Non-finite/invalid output rejection -- also leaves no output file
# --------------------------------------------------------------------------


def test_non_finite_output_rejected(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)

    def _nan_noise(waveform, *_args, **_kwargs):
        result = waveform.clone()
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(pertubation_service, "add_gaussian_noise", _nan_noise)

    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)
    assert not output.exists()


def test_empty_output_rejected(monkeypatch, source_and_output):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(pertubation_service, "add_gaussian_noise", lambda waveform, *a, **k: torch.zeros(1, 0))

    with pytest.raises(robustness.PerturbationOutputInvalid):
        robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)
    assert not output.exists()


# --------------------------------------------------------------------------
# Real (non-monkeypatched) transforms -- output file is actually written
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_type,params",
    [
        ("noise", {"noise_level": 0.05}),
        ("time_masking", {"mask_start_percent": 20, "mask_end_percent": 40}),
        ("frequency_masking", {"mask_freq_start": 500, "mask_freq_end": 2000}),
    ],
)
def test_real_transforms_apply_successfully(monkeypatch, source_and_output, perturbation_type, params):
    source, output = source_and_output
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.perturb_and_compare("ecapa-tdnn", source, perturbation_type, params, output)

    assert output.exists()
    assert "similarity" in result and "embedding_shift" in result


def test_stereo_source_is_downmixed_before_perturbation(monkeypatch, tmp_path):
    stereo_path = tmp_path / "source-stereo.wav"
    t = np.linspace(0, 1.0, 16000, False)
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 240.0 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(stereo_path, stereo, 16000)
    output = tmp_path / "output.wav"

    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.perturb_and_compare("ecapa-tdnn", stereo_path, "noise", {"noise_level": 0.05}, output)

    assert output.exists()
    assert "similarity" in result


def test_non_16khz_source_still_produces_valid_result(monkeypatch, tmp_path):
    source = _write_wav(tmp_path / "source-8k.wav", sample_rate=8000, freq=300.0)
    output = tmp_path / "output.wav"

    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _HashDerivedAdapter(ECAPA_SPEC.embedding_dimension))

    result = robustness.perturb_and_compare("ecapa-tdnn", source, "noise", {"noise_level": 0.05}, output)

    assert output.exists()
    assert "similarity" in result


# --------------------------------------------------------------------------
# Router: /tasks/verification/perturbation
# --------------------------------------------------------------------------


def _fake_router_result(**overrides) -> dict:
    result = {
        "model": ECAPA_SPEC.key,
        "model_label": ECAPA_SPEC.label,
        "threshold": ECAPA_SPEC.threshold,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05, "seed": 0}},
        "similarity": 0.8,
        "same_speaker": True,
        "embedding_shift": 0.2,
        "original_embedding": [0.1] * ECAPA_SPEC.embedding_dimension,
        "perturbed_embedding": [0.1] * ECAPA_SPEC.embedding_dimension,
    }
    result.update(overrides)
    return result


def _fake_perturb_and_compare(
    model_key, source_path, perturbation_type, perturbation_params, output_path, cached_original_embedding=None
):
    """Stand-in for `perturb_and_compare` used to test router behavior in
    isolation. Still writes a real file to `output_path`, since the router
    promotes that path into a real session asset afterward."""

    _write_wav(Path(output_path), freq=250.0)
    return _fake_router_result()


@pytest.mark.asyncio
async def test_perturbation_endpoint_demo_source_creates_session_asset(client, fake_dataset_dir):
    recording_id = fake_dataset_dir[0]
    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch("app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare):
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ecapa-tdnn"
    assert body["similarity"] == 0.8
    assert body["same_speaker"] is True
    assert body["embedding_shift"] == 0.2
    assert body["threshold"] == ECAPA_SPEC.threshold
    assert body["source_recording_id"] == recording_id
    assert body["session_asset"]["asset_id"].startswith("asset_")
    assert body["session_asset"]["origin"] == "perturbation"
    assert "original_embedding" not in body
    assert "perturbed_embedding" not in body
    _assert_no_ground_truth_leak(body)

    listing = await client.get("/tasks/verification/session-assets")
    assert listing.status_code == 200
    assert any(item["asset_id"] == body["session_asset"]["asset_id"] for item in listing.json()["assets"])


@pytest.mark.asyncio
async def test_perturbation_endpoint_session_asset_source(client):
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["asset_id"]

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": asset_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch("app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare):
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["source_recording_id"] == asset_id
    assert body["session_asset"]["asset_id"] != asset_id


@pytest.mark.asyncio
async def test_perturbation_endpoint_reperturbing_already_perturbed_asset(client):
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    asset_a = upload.json()["asset_id"]

    payload_a = {
        "model": "ecapa-tdnn",
        "recording_id": asset_a,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch("app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare):
        response_b = await client.post("/tasks/verification/perturbation", json=payload_a)
    asset_b = response_b.json()["session_asset"]["asset_id"]

    payload_b = {
        "model": "ecapa-tdnn",
        "recording_id": asset_b,
        "perturbation": {"type": "time_stretch", "params": {"stretch_factor": 1.2}},
    }
    with patch("app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare):
        response_c = await client.post("/tasks/verification/perturbation", json=payload_b)

    assert response_c.status_code == 200
    asset_c = response_c.json()["session_asset"]["asset_id"]

    metadata = await client.get(f"/tasks/verification/session-assets/{asset_c}")
    assert metadata.json()["source_asset_id"] == asset_b


@pytest.mark.asyncio
async def test_perturbation_endpoint_cross_session_asset_rejected(client):
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    asset_id = upload.json()["asset_id"]

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": asset_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other_client:
        response = await other_client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_perturbation_endpoint_unknown_asset_id_is_404(client):
    response = await client.post(
        "/tasks/verification/perturbation",
        json={
            "model": "ecapa-tdnn",
            "recording_id": "asset_0000000000000000000000000000000",
            "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_perturbation_endpoint_unknown_dataset_id_is_404(client, fake_dataset_dir):
    response = await client.post(
        "/tasks/verification/perturbation",
        json={
            "model": "ecapa-tdnn",
            "recording_id": "rec_0000000000000000",
            "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_perturbation_endpoint_unsupported_model_is_400(client, fake_dataset_dir):
    response = await client.post(
        "/tasks/verification/perturbation",
        json={
            "model": "not-a-real-model",
            "recording_id": fake_dataset_dir[0],
            "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_perturbation_endpoint_noop_perturbation_creates_no_asset(client, fake_dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("app.tasks.verification.session_assets._storage_root", lambda: tmp_path)

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": fake_dataset_dir[0],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch(
        "app.tasks.verification.router.perturb_and_compare",
        side_effect=robustness.PerturbationNotApplied("no-op"),
    ):
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 422

    sid = response.cookies.get(settings.SESSION_COOKIE_NAME)
    assert sid is not None
    session_dir = tmp_path / sid
    leftover = list(session_dir.iterdir()) if session_dir.is_dir() else []
    assert leftover == []

    listing = await client.get("/tasks/verification/session-assets")
    assert listing.json()["assets"] == []


@pytest.mark.asyncio
async def test_perturbation_endpoint_transform_failure_creates_no_asset(client, fake_dataset_dir):
    payload = {
        "model": "ecapa-tdnn",
        "recording_id": fake_dataset_dir[0],
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch(
        "app.tasks.verification.router.perturb_and_compare",
        side_effect=robustness.PerturbationOutputInvalid("bad output"),
    ):
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 422
    listing = await client.get("/tasks/verification/session-assets")
    assert listing.json()["assets"] == []


@pytest.mark.asyncio
async def test_perturbation_endpoint_original_embedding_cache_miss_stores_value(client, fake_dataset_dir):
    recording_id = fake_dataset_dir[0]
    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch(
        "app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare
    ) as mock_compare:
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    assert mock_compare.call_args.args[-1] is None  # cached_original_embedding: miss

    audio_path = dataset.resolve_recording_path(recording_id)
    audio_hash = cache.hash_audio_files([audio_path])[0]
    key = cache.embedding_cache_key(
        model_key="ecapa-tdnn",
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        audio_sha256=audio_hash,
    )
    cached = await cache.get_embedding(key, ECAPA_SPEC.embedding_dimension)
    assert cached is not None


@pytest.mark.asyncio
async def test_perturbation_endpoint_original_embedding_cache_hit_skips_recompute_flag(client, fake_dataset_dir):
    recording_id = fake_dataset_dir[0]
    audio_path = dataset.resolve_recording_path(recording_id)
    audio_hash = cache.hash_audio_files([audio_path])[0]
    key = cache.embedding_cache_key(
        model_key="ecapa-tdnn",
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        audio_sha256=audio_hash,
    )
    dim = ECAPA_SPEC.embedding_dimension
    stored = F.normalize(torch.tensor([1.0] + [0.0] * (dim - 1)), p=2, dim=0)
    await cache.set_embedding(key, "ecapa-tdnn", stored.tolist())

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch(
        "app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare
    ) as mock_compare:
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    cached_arg = mock_compare.call_args.args[-1]
    assert cached_arg is not None
    assert torch.allclose(cached_arg, stored, atol=1e-5)


@pytest.mark.asyncio
async def test_perturbation_endpoint_corrupted_cache_treated_as_miss(client, fake_dataset_dir):
    recording_id = fake_dataset_dir[0]
    audio_path = dataset.resolve_recording_path(recording_id)
    audio_hash = cache.hash_audio_files([audio_path])[0]
    key = cache.embedding_cache_key(
        model_key="ecapa-tdnn",
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        audio_sha256=audio_hash,
    )
    # Structurally valid (right length, numeric) but non-finite -- caught only
    # by `rehydrate_cached_embedding`, not by `cache.get_embedding` itself.
    await cache.set_embedding(key, "ecapa-tdnn", [float("nan")] * ECAPA_SPEC.embedding_dimension)

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch(
        "app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare
    ) as mock_compare:
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    assert mock_compare.call_args.args[-1] is None  # corrupted hit -> treated as miss
