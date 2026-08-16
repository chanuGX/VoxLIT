"""Parity tests for the shared Range/HEAD/206 audio-streaming helper: the
demo-dataset audio route and the new session-asset audio route must behave
identically, and the demo route's own behavior must be unchanged after being
refactored to use the shared helper (`stream_audio_file`).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.core.settings import settings
from app.main import app

FAKE_DEMO_FILES = ["id10018_01.wav", "id10324_01.wav"]


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 0.5, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    content = _wav_bytes()
    for filename in FAKE_DEMO_FILES:
        (dataset_dir / filename).write_bytes(content)
    return dataset_dir, content


async def _assert_streams_correctly(client, url: str, cookies, expected_bytes: bytes):
    # Plain GET -- full content, safe headers.
    response = await client.get(url, cookies=cookies)
    assert response.status_code == 200
    assert response.content == expected_bytes
    assert response.headers.get("accept-ranges") == "bytes"
    assert "inline" in response.headers.get("content-disposition", "")

    # Ranged GET -- real 206 partial content, not the whole file.
    ranged = await client.get(url, cookies=cookies, headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206
    assert ranged.content == expected_bytes[:4]
    assert ranged.headers.get("content-range") == f"bytes 0-3/{len(expected_bytes)}"
    assert ranged.headers.get("accept-ranges") == "bytes"

    # HEAD -- succeeds, no body.
    head = await client.head(url, cookies=cookies)
    assert head.status_code == 200
    assert head.content == b""


@pytest.mark.asyncio
async def test_demo_route_streams_correctly(client, fake_dataset_dir):
    _dataset_dir, content = fake_dataset_dir
    listing = await client.get("/tasks/verification/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]
    url = f"/tasks/verification/dataset/recordings/{recording_id}/audio"
    await _assert_streams_correctly(client, url, cookies=None, expected_bytes=content)


@pytest.mark.asyncio
async def test_session_asset_route_streams_identically_to_demo_route(client):
    session_response = await client.get("/session")
    cookies = session_response.cookies

    content = _wav_bytes(freq=440.0)
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", content, "audio/wav")},
        cookies=cookies,
    )
    assert upload.status_code == 200
    asset_id = upload.json()["asset_id"]

    url = f"/tasks/verification/session-assets/{asset_id}/audio"
    await _assert_streams_correctly(client, url, cookies=cookies, expected_bytes=content)


@pytest.mark.asyncio
async def test_session_asset_audio_requires_owning_session():
    # Two fully independent clients/cookie jars -- no ambiguity about which
    # session a given request actually carries.
    async with AsyncClient(app=app, base_url="http://test") as client_a:
        await client_a.get("/session")
        upload = await client_a.post(
            "/tasks/verification/session-assets/upload",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        )
        assert upload.status_code == 200
        asset_id = upload.json()["asset_id"]

    async with AsyncClient(app=app, base_url="http://test") as client_b:
        response = await client_b.get(f"/tasks/verification/session-assets/{asset_id}/audio")
        assert response.status_code == 404
