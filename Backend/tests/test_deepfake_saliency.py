"""Feature 3 — waveform-aligned saliency (SRS DF-14, DF-15).

The attribution itself is produced with a tiny stand-in model, so the tests
pin the CONTRACT and the time alignment rather than whatever a 300M-parameter
checkpoint happens to output. No real model is ever loaded.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from app.core.settings import settings
from app.tasks.deepfake import saliency

SAMPLE_RATE = 16_000


class _LoudHalfModel(torch.nn.Module):
    """Its spoof logit depends only on the SECOND half of the input.

    So the attribution must land in the second half — which makes the time
    alignment testable rather than merely plausible.
    """

    def zero_grad(self, set_to_none: bool = True):  # noqa: D102
        pass

    def forward(self, waveform):
        half = waveform.shape[-1] // 2
        spoof = (waveform[..., half:] ** 2).sum()
        return torch.stack([torch.zeros_like(spoof), spoof]).unsqueeze(0)


def _adapter():
    """A Tier-B-shaped adapter: raw waveform in, no feature extractor."""
    return SimpleNamespace(
        model=_LoudHalfModel(),
        spoof_index=1,
        analysis_window_seconds=30.0,
        _xlsr_mamba=SimpleNamespace(pad_or_tile=lambda samples: samples),
    )


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.ones(SAMPLE_RATE * 4, dtype=np.float32) * 0.5, SAMPLE_RATE)
    return path


# --- attribution ----------------------------------------------------------


def test_attribution_lands_where_the_model_actually_looked(clip):
    """DF-14 — the heat has to be in the right place in TIME."""
    attribution, seconds = saliency._attribution_over_time(_adapter(), clip, 30.0)

    assert seconds == pytest.approx(4.0, abs=0.01)
    half = attribution.size // 2
    assert attribution[:half].sum() == 0.0
    assert attribution[half:].sum() > 0.0


def test_segments_cover_the_clip_end_to_end(clip):
    attribution, seconds = saliency._attribution_over_time(_adapter(), clip, 30.0)

    segments, series = saliency._to_segments(attribution, seconds, 20)

    assert len(segments) == len(series) == 20
    assert segments[0]["start_time"] == 0.0
    assert segments[-1]["end_time"] == pytest.approx(seconds, abs=0.01)
    for previous, following in zip(segments, segments[1:]):
        assert previous["end_time"] == pytest.approx(following["start_time"], abs=1e-6)


def test_segments_match_the_shared_saliency_service_contract(clip):
    """DF-14 says to reuse the shared visualisation, so the shape must match."""
    attribution, seconds = saliency._attribution_over_time(_adapter(), clip, 30.0)

    segments, _ = saliency._to_segments(attribution, seconds, 10)

    assert set(segments[0]) == {"start_time", "end_time", "saliency", "intensity"}


def test_attribution_is_normalised_within_the_clip(clip):
    attribution, seconds = saliency._attribution_over_time(_adapter(), clip, 30.0)

    _, series = saliency._to_segments(attribution, seconds, 20)

    assert max(series) == pytest.approx(1.0)
    assert min(series) >= 0.0


def test_a_flat_attribution_does_not_divide_by_zero():
    segments, series = saliency._to_segments(np.zeros(1000, dtype=np.float32), 2.0, 10)

    assert len(segments) == 10
    assert all(value == 0.0 for value in series)


# --- duration caps (DF-15) ------------------------------------------------


def test_the_shared_saliency_cap_is_applied(tmp_path):
    """DF-15 — the same cap the shared service uses."""
    path = tmp_path / "long.wav"
    sf.write(path, np.ones(SAMPLE_RATE * 30, dtype=np.float32) * 0.5, SAMPLE_RATE)

    _, seconds = saliency._attribution_over_time(_adapter(), path, 5.0)

    assert seconds == pytest.approx(5.0, abs=0.01)


def test_the_models_own_window_wins_when_it_is_tighter(clip):
    """AST looks at 10.24 s regardless of what the shared cap allows."""
    adapter = _adapter()
    adapter.analysis_window_seconds = 1.5

    _, seconds = saliency._attribution_over_time(adapter, clip, 30.0)

    assert seconds == pytest.approx(1.5, abs=0.01)


# --- endpoint -------------------------------------------------------------

PROTOCOL_ID = "LA_E_5000001"


@pytest.fixture
def saliency_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    # Silence, then a tone: gives the overlay a real speech region to find.
    quiet = np.random.RandomState(0).randn(SAMPLE_RATE).astype(np.float32) * 1e-4
    t = np.linspace(0, 2, SAMPLE_RATE * 2, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(audio_dir / f"{PROTOCOL_ID}.flac", np.concatenate([quiet, tone]), SAMPLE_RATE)
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        f"LA_0069 {PROTOCOL_ID} - - bonafide\n"
    )
    return audio_dir


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(saliency, "get_model", lambda key: _adapter())


async def test_saliency_endpoint_returns_the_shared_contract(
    client, saliency_dataset, stub_model
):
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    response = await client.post(
        "/tasks/deepfake/saliency",
        json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert {"model", "method", "segments", "total_duration", "series"} <= set(payload)
    assert payload["segments"]
    assert len(payload["series"]) == len(payload["segments"])


async def test_saliency_names_its_method(client, saliency_dataset, stub_model):
    """DF-15 — the method is identified, not implied."""
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    payload = (
        await client.post(
            "/tasks/deepfake/saliency",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    assert payload["method"] == saliency.METHOD
    assert payload["method_label"] == saliency.METHOD_LABEL
    assert payload["target"] == "spoof logit"
    assert payload["max_saliency_seconds"] == saliency.MAX_SALIENCY_SECONDS


async def test_saliency_reports_where_the_speech_is(client, saliency_dataset, stub_model):
    """The overlay is what lets a reader tell heat-on-voice from heat-on-silence."""
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    payload = (
        await client.post(
            "/tasks/deepfake/saliency",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    assert payload["speech_intervals"]
    start, end = payload["speech_intervals"][0]
    assert start == pytest.approx(1.0, abs=0.2)  # tone begins after 1 s of quiet
    assert 0.0 <= payload["saliency_in_speech_fraction"] <= 1.0


async def test_saliency_rejects_an_unknown_model(client, saliency_dataset):
    response = await client.post(
        "/tasks/deepfake/saliency", json={"model": "nope", "recording_id": "rec_x"}
    )

    assert response.status_code == 400


async def test_saliency_404s_on_an_unknown_recording(client, saliency_dataset):
    response = await client.post(
        "/tasks/deepfake/saliency",
        json={"model": "xlsr-deepfake", "recording_id": "rec_missing"},
    )

    assert response.status_code == 404


async def test_saliency_is_cached(client, saliency_dataset, stub_model):
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]
    body = {"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]}

    first = (await client.post("/tasks/deepfake/saliency", json=body)).json()
    second = (await client.post("/tasks/deepfake/saliency", json=body)).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert first["series"] == second["series"]


async def test_saliency_leaks_no_ground_truth(client, saliency_dataset, stub_model):
    recordings = (await client.get("/tasks/deepfake/dataset/recordings")).json()["recordings"]

    payload = (
        await client.post(
            "/tasks/deepfake/saliency",
            json={"model": "xlsr-deepfake", "recording_id": recordings[0]["recording_id"]},
        )
    ).json()

    assert PROTOCOL_ID not in json.dumps(payload)
