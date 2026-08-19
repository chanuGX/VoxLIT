"""Feature 2 — the silence and non-speech probe (SRS DF-10..DF-12).

The segmentation is tested on synthesised audio with silence in known
places, so the assertions are about where the boundaries land rather than
about whatever a real clip happens to contain. `run_detection` is stubbed;
no model is ever loaded.
"""

from __future__ import annotations

import json
from importlib import import_module

import numpy as np
import pytest
import soundfile as sf

from app.core.settings import settings
from app.tasks.deepfake import silence_probe

deepfake_router = import_module("app.tasks.deepfake.router")

SAMPLE_RATE = 16_000


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """A loud 220 Hz tone — stands in for speech."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    """Near-silence: low-level noise, well under the 30 dB threshold."""
    rng = np.random.RandomState(0)
    return (rng.randn(int(SAMPLE_RATE * seconds)) * 1e-4).astype(np.float32)


# --- segmentation (DF-11) -------------------------------------------------


def test_speech_is_found_between_the_silences():
    waveform = np.concatenate([quiet(1.0), tone(2.0), quiet(1.0)])

    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    assert len(segmentation.speech_intervals) == 1
    start, end = segmentation.speech_intervals[0]
    # librosa works in frames, so allow a frame or two of slack.
    assert start / SAMPLE_RATE == pytest.approx(1.0, abs=0.1)
    assert end / SAMPLE_RATE == pytest.approx(3.0, abs=0.1)


def test_non_speech_duration_counts_internal_pauses_too():
    """DF-10 says "the non-speech portions", not only the outer ones."""
    waveform = np.concatenate([quiet(0.5), tone(1.0), quiet(1.0), tone(1.0), quiet(0.5)])

    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    assert len(segmentation.speech_intervals) == 2
    # 0.5 leading + 1.0 internal + 0.5 trailing = 2.0s nominal. librosa works
    # at ~32 ms frame resolution, so each of the four boundaries costs up to a
    # frame; the tolerance is sized for that, not fitted to the result.
    assert segmentation.seconds(segmentation.non_speech_samples) == pytest.approx(2.0, abs=0.35)


def test_the_threshold_is_relative_to_the_clips_own_level():
    """A quietly recorded clip must segment the same as a loud one.

    An absolute noise floor would find no speech at all in the quiet copy.
    """
    loud = np.concatenate([quiet(0.5), tone(1.0, amplitude=0.9), quiet(0.5)])
    soft = np.concatenate([quiet(0.5) * 0.01, tone(1.0, amplitude=0.009), quiet(0.5) * 0.01])

    loud_segmentation = silence_probe.segment_speech(loud, SAMPLE_RATE)
    soft_segmentation = silence_probe.segment_speech(soft, SAMPLE_RATE)

    assert len(loud_segmentation.speech_intervals) == len(soft_segmentation.speech_intervals) == 1
    assert loud_segmentation.speech_intervals[0][0] == pytest.approx(
        soft_segmentation.speech_intervals[0][0], abs=SAMPLE_RATE * 0.1
    )


# --- the two ablations ----------------------------------------------------


def test_trimming_removes_the_outer_silence_only():
    waveform = np.concatenate([quiet(1.0), tone(2.0), quiet(1.0)])
    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    trimmed, _ = silence_probe.build_variants(waveform, segmentation)

    assert len(trimmed) / SAMPLE_RATE == pytest.approx(2.0, abs=0.2)


def test_the_non_speech_variant_keeps_only_the_silence():
    waveform = np.concatenate([quiet(1.0), tone(2.0), quiet(1.0)])
    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    _, non_speech = silence_probe.build_variants(waveform, segmentation)

    assert len(non_speech) / SAMPLE_RATE == pytest.approx(2.0, abs=0.2)
    # Whatever is left must be quiet — if speech leaked in, the probe lies.
    assert float(np.abs(non_speech).max()) < 0.05


def test_the_two_variants_partition_the_clip():
    """Trimmed + non-speech should account for the whole clip.

    (Internal pauses appear in both, so this is an inequality, not equality
    — but nothing may be invented.)
    """
    waveform = np.concatenate([quiet(0.5), tone(1.0), quiet(0.8), tone(1.0), quiet(0.5)])
    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    trimmed, non_speech = silence_probe.build_variants(waveform, segmentation)

    assert len(trimmed) <= len(waveform)
    assert len(non_speech) <= len(waveform)
    assert segmentation.speech_samples + segmentation.non_speech_samples == len(waveform)


def test_an_entirely_silent_clip_reads_as_speech_and_df12_catches_it():
    """The consequence of a RELATIVE threshold, pinned deliberately.

    DF-11 requires the threshold to be relative to the clip's own level. In a
    clip that is nothing but noise, the noise IS the peak, so librosa marks it
    all as "speech" and no non-speech is found. That is not a failure of the
    segmentation — it is why DF-12 exists, and this test pins the handover
    between the two requirements.
    """
    waveform = quiet(2.0)
    segmentation = silence_probe.segment_speech(waveform, SAMPLE_RATE)

    non_speech_seconds = segmentation.seconds(segmentation.non_speech_samples)

    assert non_speech_seconds < silence_probe.MIN_NON_SPEECH_SECONDS


# --- endpoint -------------------------------------------------------------

PROTOCOL_ID = "LA_E_4000001"


@pytest.fixture
def probe_dataset(monkeypatch, tmp_path):
    """A real WAV with 1s silence, 2s tone, 1s silence."""
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    waveform = np.concatenate([quiet(1.0), tone(2.0), quiet(1.0)])
    sf.write(audio_dir / f"{PROTOCOL_ID}.flac", waveform, SAMPLE_RATE)
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        f"LA_0069 {PROTOCOL_ID} - - bonafide\n"
    )
    return audio_dir


@pytest.fixture
def stub_detection(monkeypatch):
    """Score by duration, so the three variants get distinguishable values."""
    calls = []

    def _fake_run_detection(model_key, audio_path):
        import soundfile as sf_read

        samples, rate = sf_read.read(audio_path)
        seconds = len(samples) / rate
        calls.append(round(seconds, 2))
        score = 0.9 if seconds > 2.5 else 0.2
        return {
            "model": model_key,
            "model_label": "wav2vec2 XLS-R (Model A)",
            "model_id": "stub",
            "decision": "spoof" if score >= 0.5 else "bonafide",
            "threshold": 0.5,
            "threshold_calibrated": False,
            "threshold_version": "deepfake-threshold-v0-uncalibrated",
            "spoof_probability": score,
            "bonafide_probability": 1 - score,
        }

    monkeypatch.setattr(silence_probe, "run_detection", _fake_run_detection)
    return calls


async def test_probe_returns_three_scores_together(client, probe_dataset, stub_detection):
    """SRS DF-10."""
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]
    recording_id = recordings[0]["recording_id"]

    response = await client.post(
        "/tasks/deepfake/silence-probe",
        json={"model": "xlsr-deepfake", "recording_id": recording_id},
    )

    assert response.status_code == 200
    variants = response.json()["variants"]
    assert set(variants) == {"original", "trimmed", "non_speech"}
    for variant in variants.values():
        assert variant["applicable"] is True
        assert variant["spoof_probability"] is not None
    # Three forward passes, one per variant.
    assert len(stub_detection) == 3


async def test_probe_reports_the_threshold_it_used(client, probe_dataset, stub_detection):
    """SRS DF-11 — the threshold travels with the result."""
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    payload = (
        await client.post(
            "/tasks/deepfake/silence-probe",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    assert payload["silence_top_db"] == silence_probe.SILENCE_TOP_DB
    assert payload["non_speech_seconds"] == pytest.approx(2.0, abs=0.2)
    assert payload["speech_seconds"] == pytest.approx(2.0, abs=0.2)


async def test_probe_reports_not_applicable_when_there_is_too_little_silence(
    client, monkeypatch, tmp_path, stub_detection
):
    """SRS DF-12 — refuse rather than report an unreliable silence score."""
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    # Speech end to end: nothing to keep in the non-speech variant.
    sf.write(audio_dir / "LA_E_4000002.flac", tone(3.0), SAMPLE_RATE)
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "LA_0069 LA_E_4000002 - - bonafide\n"
    )

    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]
    payload = (
        await client.post(
            "/tasks/deepfake/silence-probe",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    non_speech = payload["variants"]["non_speech"]
    assert non_speech["applicable"] is False
    assert non_speech["spoof_probability"] is None
    assert "non-speech" in non_speech["reason"]
    # The original must still have been scored.
    assert payload["variants"]["original"]["applicable"] is True


async def test_probe_rejects_an_unknown_model(client, probe_dataset):
    response = await client.post(
        "/tasks/deepfake/silence-probe", json={"model": "nope", "recording_id": "rec_x"}
    )

    assert response.status_code == 400


async def test_probe_404s_on_an_unknown_recording(client, probe_dataset):
    response = await client.post(
        "/tasks/deepfake/silence-probe",
        json={"model": "xlsr-deepfake", "recording_id": "rec_does_not_exist"},
    )

    assert response.status_code == 404


async def test_probe_is_cached_on_the_second_call(client, probe_dataset, stub_detection):
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]
    body = {"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]}

    first = (await client.post("/tasks/deepfake/silence-probe", json=body)).json()
    calls_after_first = len(stub_detection)
    second = (await client.post("/tasks/deepfake/silence-probe", json=body)).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert len(stub_detection) == calls_after_first  # no further inference


async def test_probe_leaks_no_ground_truth(client, probe_dataset, stub_detection):
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    payload = (
        await client.post(
            "/tasks/deepfake/silence-probe",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    body = json.dumps(payload)
    assert PROTOCOL_ID not in body
    assert "bonafide" not in body.replace('"decision": "bonafide"', "")
