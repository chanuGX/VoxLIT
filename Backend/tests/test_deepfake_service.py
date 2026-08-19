"""Unit tests for the deepfake service's model registry and label mapping.

None of these load a checkpoint. The spoof-index resolver is the piece worth
testing hardest: if it picks the wrong class, every score the task produces is
inverted and nothing downstream can tell.
"""

import numpy as np
import pytest
import soundfile as sf

from app.tasks.deepfake import service


def _write_tone(path, sample_rate=48_000, seconds=0.5, channels=1, fmt=None):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    if channels > 1:
        tone = np.stack([tone] * channels, axis=-1)
    sf.write(path, tone, sample_rate, format=fmt)
    return path


# --- audio loading -------------------------------------------------------
# Regression cover for a real break: torchaudio >= 2.9 delegates `load` to
# TorchCodec, which is not installed, so the copied `torchaudio.load` loader
# raised ImportError on every clip. Decoding now goes through soundfile.


@pytest.mark.parametrize("fmt, suffix", [("FLAC", ".flac"), ("WAV", ".wav")])
def test_load_waveform_decodes_both_dataset_and_upload_formats(tmp_path, fmt, suffix):
    """ASVspoof ships FLAC; uploads are WAV. Both must decode."""
    path = _write_tone(tmp_path / f"clip{suffix}", fmt=fmt)

    waveform, sample_rate = service._load_waveform(path)

    assert sample_rate == service.TARGET_SAMPLE_RATE
    assert waveform.shape[0] == 1  # mono
    assert waveform.dtype.is_floating_point


def test_load_waveform_resamples_to_16k(tmp_path):
    path = _write_tone(tmp_path / "clip.flac", sample_rate=48_000, seconds=1.0, fmt="FLAC")

    waveform, sample_rate = service._load_waveform(path)

    assert sample_rate == 16_000
    # 1s at 48 kHz -> ~16000 samples after resampling.
    assert abs(waveform.shape[1] - 16_000) < 100


def test_load_waveform_downmixes_to_mono(tmp_path):
    path = _write_tone(tmp_path / "stereo.wav", channels=2, fmt="WAV")

    waveform, _ = service._load_waveform(path)

    assert waveform.shape[0] == 1


def test_load_waveform_rejects_a_non_audio_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")

    with pytest.raises(ValueError):
        service._load_waveform(path)


def test_model_id_avoids_the_shared_saliency_dispatch_substring():
    """`saliency_service.detect_model_type` matches on `"wav2vec" in model`.

    A deepfake model id carrying that substring would silently route to the
    EMOTION model's saliency and return emotion attributions labelled as
    deepfake attributions, so the id must not contain it.
    """
    for key in service.MODEL_SPECS:
        assert "wav2vec" not in key.lower()
        assert "whisper" not in key.lower()


def test_get_model_spec_returns_the_registered_spec():
    spec = service.get_model_spec("xlsr-deepfake")

    assert spec.model_id == "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
    assert spec.sampling_rate == 16_000


def test_get_model_spec_rejects_unknown_model():
    with pytest.raises(service.UnsupportedDeepfakeModel):
        service.get_model_spec("not-a-model")


def test_both_phase_one_models_are_registered():
    assert set(service.MODEL_SPECS) == {"xlsr-deepfake", "ast-fakeaudio"}

    ast = service.get_model_spec("ast-fakeaudio")
    assert ast.model_id == "WpythonW/ast-fakeaudio-detector"
    assert ast.gated is True  # requires accepting conditions + HF_TOKEN
    assert service.get_model_spec("xlsr-deepfake").gated is False


# --- analysis window ------------------------------------------------------
# AST pads/truncates to a fixed 1024-frame (10.24s) window and does it
# silently, so the reported `analysed_seconds` must come from the extractor,
# never from the clip's duration.


class _FakeASTExtractor:
    max_length = 1024
    num_mel_bins = 128


class _FakeWav2Vec2Extractor:
    """No max_length, no mel bins — consumes whatever it is given."""


def test_analysis_window_follows_ast_fixed_input_length():
    window = service._analysis_window_seconds(_FakeASTExtractor())

    assert window == pytest.approx(10.24)


def test_analysis_window_falls_back_to_our_cap_for_waveform_models():
    window = service._analysis_window_seconds(_FakeWav2Vec2Extractor())

    assert window == service.MAX_ANALYSIS_SECONDS


def test_analysis_window_never_exceeds_our_defensive_cap(monkeypatch):
    """A checkpoint asking for a huge window must not blow past the cap."""

    class _Huge:
        max_length = 100_000  # 1000 s
        num_mel_bins = 128

    assert service._analysis_window_seconds(_Huge()) == service.MAX_ANALYSIS_SECONDS


# --- load failures --------------------------------------------------------


def test_gated_repo_failure_message_is_actionable():
    spec = service.get_model_spec("ast-fakeaudio")

    message = service._load_failure_message(spec, OSError("401 Client Error"))

    assert "gated" in message.lower()
    assert "HF_TOKEN" in message
    assert spec.model_id in message


def test_ungated_failure_message_stays_plain():
    spec = service.get_model_spec("xlsr-deepfake")

    message = service._load_failure_message(spec, OSError("disk on fire"))

    assert "disk on fire" in message
    assert "HF_TOKEN" not in message


def test_a_gated_looking_error_is_detected_even_on_an_ungated_spec():
    """Repos can become gated after we recorded them as public."""
    spec = service.get_model_spec("xlsr-deepfake")

    message = service._load_failure_message(spec, OSError("Access to model X is restricted"))

    assert "HF_TOKEN" in message


def test_threshold_is_declared_uncalibrated():
    """Until Feature 1 runs an EER sweep, nothing may imply calibration."""
    spec = service.get_model_spec("xlsr-deepfake")

    assert spec.threshold_calibrated is False
    assert "uncalibrated" in service.THRESHOLD_VERSION


def test_list_models_is_serializable():
    models = service.list_models()

    assert len(models) == len(service.MODEL_SPECS)
    assert models[0]["key"] == "xlsr-deepfake"


@pytest.mark.parametrize(
    "id2label, expected",
    [
        ({0: "bonafide", 1: "spoof"}, 1),
        ({0: "spoof", 1: "bonafide"}, 0),
        ({0: "REAL", 1: "FAKE"}, 1),
        ({0: "fake", 1: "real"}, 0),
        ({0: "bona fide", 1: "spoofed"}, 1),
        ({0: "genuine_audio", 1: "deepfake_audio"}, 1),
        # Only one side is nameable — the other follows by elimination.
        ({0: "LABEL_0", 1: "spoof"}, 1),
        ({0: "bonafide", 1: "LABEL_1"}, 1),
        # String keys, as they arrive from a JSON config.
        ({"0": "bonafide", "1": "spoof"}, 1),
    ],
)
def test_resolve_spoof_index(id2label, expected):
    assert service.resolve_spoof_index(id2label) == expected


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "LABEL_0", 1: "LABEL_1"},          # nothing nameable
        {0: "spoof", 1: "fake"},               # both look like spoof
        {0: "real", 1: "bonafide"},            # both look like bona fide
    ],
)
def test_resolve_spoof_index_refuses_to_guess(id2label):
    with pytest.raises(service.DeepfakeModelUnavailable) as excinfo:
        service.resolve_spoof_index(id2label)

    # The error must name what it saw, so the mapping can be fixed by hand.
    assert str(id2label[0]) in str(excinfo.value)


@pytest.mark.parametrize(
    "id2label",
    [{0: "bonafide"}, {0: "bonafide", 1: "spoof", 2: "unknown"}, {}],
)
def test_resolve_spoof_index_requires_a_binary_head(id2label):
    with pytest.raises(service.DeepfakeModelUnavailable):
        service.resolve_spoof_index(id2label)
