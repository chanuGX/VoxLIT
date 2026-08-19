"""Endpoint tests for /tasks/deepfake.

`run_detection` is monkeypatched everywhere below — these tests must never
download the 1.2 GB checkpoint. What is being tested is the routing, the error
mapping and the cache-through behaviour, not the model.
"""

import json
from importlib import import_module

import pytest

from app.core.settings import settings

# NOT `from app.tasks.deepfake import router` -- the package __init__ does
# `from .router import router`, so that name is the APIRouter object, not the
# module, and monkeypatching it would silently patch the wrong thing.
deepfake_router = import_module("app.tasks.deepfake.router")

FAKE_PROTOCOL = {
    "LA_E_2000001": ("-", "bonafide"),
    "LA_E_2000002": ("A10", "spoof"),
}

FAKE_DETECTION = {
    "model": "xlsr-deepfake",
    "model_label": "wav2vec2 XLS-R (Model A)",
    "model_id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "decision": "spoof",
    "threshold": 0.5,
    "threshold_calibrated": False,
    "threshold_version": "deepfake-threshold-v0-uncalibrated",
    "spoof_probability": 0.91,
    "bonafide_probability": 0.09,
    "logits": [-1.2, 1.1],
    "id2label": {0: "bonafide", 1: "spoof"},
    "spoof_index": 1,
    "duration": 3.2,
    "analysed_seconds": 3.2,
    "truncated": False,
}


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    for file_id in FAKE_PROTOCOL:
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC-fake-audio")
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {system_id} {key}"
            for file_id, (system_id, key) in FAKE_PROTOCOL.items()
        )
        + "\n"
    )
    return audio_dir


@pytest.fixture
def stub_detection(monkeypatch):
    """Replace the model call; record how many times it actually ran."""
    calls = []

    def _fake_run_detection(model_key, audio_path):
        calls.append((model_key, str(audio_path)))
        return dict(FAKE_DETECTION)

    monkeypatch.setattr(deepfake_router, "run_detection", _fake_run_detection)
    return calls


async def test_task_is_registered(client):
    response = await client.get("/tasks")

    assert response.status_code == 200
    body = json.dumps(response.json())
    assert "deepfake" in body
    assert "task-c" not in body


async def test_models_endpoint_lists_model_a(client):
    response = await client.get("/tasks/deepfake/models")

    assert response.status_code == 200
    keys = [m["key"] for m in response.json()["models"]]
    assert "xlsr-deepfake" in keys


async def test_dataset_endpoint_reports_unavailable_without_raising(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path / "nowhere")

    response = await client.get("/tasks/deepfake/dataset")

    assert response.status_code == 200
    assert response.json()["available"] is False


async def test_recordings_listing_leaks_no_ground_truth(client, fake_dataset):
    response = await client.get("/tasks/deepfake/dataset/recordings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_recordings"] == len(FAKE_PROTOCOL)

    # The dataset id itself contains "spoof" (asvspoof2019-la) — strip that
    # known-safe token rather than weakening the check.
    body = json.dumps(payload).replace("asvspoof2019-la", "")
    assert "bonafide" not in body
    assert "spoof" not in body
    assert "A10" not in body


async def test_recordings_listing_404s_when_dataset_absent(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path / "nowhere")

    response = await client.get("/tasks/deepfake/dataset/recordings")

    assert response.status_code == 404


async def test_recording_audio_is_served(client, fake_dataset):
    listing = await client.get("/tasks/deepfake/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.get(
        f"/tasks/deepfake/dataset/recordings/{recording_id}/audio"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/flac"


async def test_run_rejects_unknown_model(client, fake_dataset):
    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "not-a-model", "recording_id": "rec_deadbeefdeadbeef"},
    )

    assert response.status_code == 400


async def test_run_404s_on_unknown_recording(client, fake_dataset):
    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "xlsr-deepfake", "recording_id": "rec_deadbeefdeadbeef"},
    )

    assert response.status_code == 404


async def test_run_returns_a_score_and_then_caches_it(
    client, fake_dataset, stub_detection
):
    listing = await client.get("/tasks/deepfake/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    first = await client.post("/tasks/deepfake/run", json=body)
    assert first.status_code == 200
    payload = first.json()
    assert payload["cached"] is False
    assert payload["recording_id"] == recording_id
    assert payload["decision"] == "spoof"
    assert payload["threshold_calibrated"] is False

    second = await client.post("/tasks/deepfake/run", json=body)
    assert second.status_code == 200
    assert second.json()["cached"] is True

    # The model ran once, not twice.
    assert len(stub_detection) == 1


async def test_run_surfaces_an_unloadable_model_as_503(
    client, fake_dataset, monkeypatch
):
    def _raise(model_key, audio_path):
        raise deepfake_router.DeepfakeModelUnavailable("no weights here")

    monkeypatch.setattr(deepfake_router, "run_detection", _raise)

    listing = await client.get("/tasks/deepfake/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "xlsr-deepfake", "recording_id": recording_id},
    )

    assert response.status_code == 503
