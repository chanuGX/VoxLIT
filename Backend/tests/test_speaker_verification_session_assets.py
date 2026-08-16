"""Focused tests for the secure session asset registry (backend foundation
commit): atomic creation/rollback, session isolation, sliding TTL on both
the asset and its session index together, sweep resilience/idempotency/
locking, and delete ordering.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from redis.exceptions import RedisError

from app.core import redis as redis_module
from app.tasks.verification import session_assets


def _write_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 0.5, freq: float = 220.0) -> Path:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return path


def _valid_sid() -> str:
    return secrets.token_hex(16)  # 32 lowercase hex chars, matches ensure_session()'s .hex format


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    """Every test in this file gets its own storage root, so tests never
    touch the real Backend/uploads/verification_sessions directory."""

    root = tmp_path / "verification_sessions"
    monkeypatch.setattr(session_assets, "_storage_root", lambda: root)
    return root


async def _create_asset(sid: str, *, origin: str = "upload") -> dict:
    temp_path = session_assets.begin_asset_write(sid, ".wav")
    _write_wav(temp_path)
    return await session_assets.promote_asset(temp_path, sid, origin=origin, extension=".wav")


def _files_in_session_dir(root: Path, sid: str) -> list[Path]:
    session_dir = root / sid
    if not session_dir.is_dir():
        return []
    return list(session_dir.iterdir())


# ---------------------------------------------------------------------------
# Round trip, isolation, sid validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_resolve_round_trip():
    sid = _valid_sid()
    metadata = await _create_asset(sid)

    assert metadata["asset_id"].startswith("asset_")
    assert metadata["origin"] == "upload"
    assert metadata["extension"] == ".wav"
    assert metadata["size_bytes"] > 0
    assert metadata["duration_seconds"] > 0

    resolved_path = await session_assets.resolve_session_asset_path(metadata["asset_id"], sid)
    assert resolved_path.exists()

    fetched = await session_assets.get_session_asset_metadata(metadata["asset_id"], sid)
    assert fetched == metadata


@pytest.mark.asyncio
async def test_cross_session_isolation():
    sid_a = _valid_sid()
    sid_b = _valid_sid()
    metadata = await _create_asset(sid_a)

    with pytest.raises(session_assets.SessionAssetNotFound):
        await session_assets.resolve_session_asset_path(metadata["asset_id"], sid_b)

    assert await session_assets.get_session_asset_metadata(metadata["asset_id"], sid_b) is None


@pytest.mark.parametrize(
    "bad_sid",
    [
        "not-a-session-id",
        "",
        "A" * 32,  # uppercase hex -- must be rejected, not silently lowercased
        "12345678-1234-5678-1234-567812345678",  # dashed uuid form
        "urn:uuid:12345678123456781234567812345678",  # urn-prefixed form
        "{12345678123456781234567812345678}",  # braced form
        "0" * 31,  # one char short
        "0" * 33,  # one char long
    ],
)
def test_sid_validation_rejects_non_exact_forms(bad_sid):
    with pytest.raises(session_assets.InvalidSessionId):
        session_assets.validate_and_canonicalize_sid(bad_sid)


def test_sid_validation_accepts_exact_hex_form():
    sid = _valid_sid()
    assert session_assets.validate_and_canonicalize_sid(sid) == sid


# ---------------------------------------------------------------------------
# Atomic creation + rollback (not just RedisError) + transactional write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promote_asset_rolls_back_on_torchaudio_info_failure(monkeypatch, isolated_storage):
    sid = _valid_sid()
    temp_path = session_assets.begin_asset_write(sid, ".wav")
    _write_wav(temp_path)

    def _raise_info(_path):
        raise RuntimeError("simulated corrupt audio")

    monkeypatch.setattr(session_assets.torchaudio, "info", _raise_info)

    with pytest.raises(RuntimeError):
        await session_assets.promote_asset(temp_path, sid, origin="upload", extension=".wav")

    assert _files_in_session_dir(isolated_storage, sid) == []
    assert not temp_path.exists()


class _RaisingPipeline:
    def set(self, *args, **kwargs):
        return self

    def sadd(self, *args, **kwargs):
        return self

    def expire(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
        return self

    def srem(self, *args, **kwargs):
        return self

    async def execute(self):
        raise RedisError("simulated redis outage")


@pytest.mark.asyncio
async def test_promote_asset_rolls_back_and_stays_transactional_on_redis_failure(monkeypatch, isolated_storage):
    sid = _valid_sid()
    temp_path = session_assets.begin_asset_write(sid, ".wav")
    _write_wav(temp_path)

    monkeypatch.setattr(redis_module.redis, "pipeline", lambda transaction=True: _RaisingPipeline())

    with pytest.raises(RedisError):
        await session_assets.promote_asset(temp_path, sid, origin="upload", extension=".wav")

    # No file survives with a final name, and neither Redis key exists --
    # proving the record and its index entry never partially applied.
    assert _files_in_session_dir(isolated_storage, sid) == []
    assert not temp_path.exists()
    assert await redis_module.redis.keys("verify:asset:*") == []


@pytest.mark.asyncio
async def test_no_storage_path_field_in_stored_record():
    """Schema assertion: the JSON payload actually written to Redis never
    contains a filesystem-path-shaped field."""

    sid = _valid_sid()
    metadata = await _create_asset(sid)

    import json

    raw = await redis_module.redis.get(session_assets._asset_key(sid, metadata["asset_id"]))
    record = json.loads(raw)
    assert "storage_path" not in record
    assert not any("path" in field.lower() for field in record.keys())


# ---------------------------------------------------------------------------
# TTL: asset + index always refreshed together, never independently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolving_refreshes_both_asset_and_index_ttl():
    sid = _valid_sid()
    metadata = await _create_asset(sid)
    asset_id = metadata["asset_id"]

    asset_key = session_assets._asset_key(sid, asset_id)
    index_key = session_assets._index_key(sid)

    # Simulate both having decayed toward expiry.
    await redis_module.redis.expire(asset_key, 50)
    await redis_module.redis.expire(index_key, 50)
    assert await redis_module.redis.ttl(asset_key) <= 50
    assert await redis_module.redis.ttl(index_key) <= 50

    await session_assets.get_session_asset_metadata(asset_id, sid)

    assert await redis_module.redis.ttl(asset_key) > 3600
    assert await redis_module.redis.ttl(index_key) > 3600


@pytest.mark.asyncio
async def test_touching_one_asset_does_not_extend_an_unrelated_asset_ttl():
    sid = _valid_sid()
    first = await _create_asset(sid)
    second = await _create_asset(sid)

    first_key = session_assets._asset_key(sid, first["asset_id"])
    await redis_module.redis.expire(first_key, 100)

    # Touch only the second asset.
    await session_assets.get_session_asset_metadata(second["asset_id"], sid)

    # The first asset's TTL must be unaffected by unrelated activity.
    assert await redis_module.redis.ttl(first_key) <= 100


# ---------------------------------------------------------------------------
# Sweep: resilience, idempotency, lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tick_survives_redis_failure_and_retries(monkeypatch, caplog):
    sid = _valid_sid()
    metadata = await _create_asset(sid)
    asset_key = session_assets._asset_key(sid, metadata["asset_id"])

    # Directly delete the Redis record (simulating expiry) so the file
    # becomes an orphan the sweep should find.
    await redis_module.redis.delete(asset_key)

    original_exists = redis_module.redis.exists

    async def _raise(*args, **kwargs):
        raise RedisError("simulated outage")

    monkeypatch.setattr(redis_module.redis, "exists", _raise)
    await session_assets._sweep_tick()  # must not raise
    await redis_module.redis.delete(session_assets._SWEEP_LOCK_KEY)

    # Restore only the patched method -- not the isolated_storage patch --
    # then confirm the very next tick retries and actually succeeds.
    monkeypatch.setattr(redis_module.redis, "exists", original_exists)
    await session_assets._sweep_tick()
    assert not (session_assets._storage_root() / sid / f"{metadata['asset_id']}.wav").exists()


@pytest.mark.asyncio
async def test_sweep_deletes_orphan_and_is_idempotent(isolated_storage):
    sid = _valid_sid()
    metadata = await _create_asset(sid)
    asset_key = session_assets._asset_key(sid, metadata["asset_id"])
    await redis_module.redis.delete(asset_key)

    deleted_first = await session_assets.sweep_expired_session_assets_once()
    assert deleted_first == 1
    assert _files_in_session_dir(isolated_storage, sid) == []
    assert metadata["asset_id"] not in await redis_module.redis.smembers(session_assets._index_key(sid))

    deleted_second = await session_assets.sweep_expired_session_assets_once()
    assert deleted_second == 0


@pytest.mark.asyncio
async def test_sweep_leaves_live_assets_alone():
    sid = _valid_sid()
    metadata = await _create_asset(sid)
    deleted = await session_assets.sweep_expired_session_assets_once()
    assert deleted == 0
    assert (await session_assets.get_session_asset_metadata(metadata["asset_id"], sid)) is not None


@pytest.mark.asyncio
async def test_sweep_tick_skips_when_lock_already_held(monkeypatch):
    calls = []

    async def _fake_sweep():
        calls.append(1)
        return 0

    monkeypatch.setattr(session_assets, "sweep_expired_session_assets_once", _fake_sweep)

    await redis_module.redis.set(
        session_assets._SWEEP_LOCK_KEY, "1", nx=True, px=session_assets.SWEEP_LOCK_MS
    )
    await session_assets._sweep_tick()
    assert calls == []  # lock held elsewhere -> this tick does nothing

    await redis_module.redis.delete(session_assets._SWEEP_LOCK_KEY)
    await session_assets._sweep_tick()
    assert calls == [1]


# ---------------------------------------------------------------------------
# Listing self-heal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listing_self_heals_stale_index_entry():
    sid = _valid_sid()
    live = await _create_asset(sid)
    stale = await _create_asset(sid)

    # Expire only the record, leaving a stale index entry.
    await redis_module.redis.delete(session_assets._asset_key(sid, stale["asset_id"]))

    listed = await session_assets.list_session_assets(sid)
    assert [item["asset_id"] for item in listed] == [live["asset_id"]]

    remaining_ids = await redis_module.redis.smembers(session_assets._index_key(sid))
    assert stale["asset_id"] not in remaining_ids


# ---------------------------------------------------------------------------
# Delete ordering + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_redis_before_file_and_survives_unlink_failure(monkeypatch):
    sid = _valid_sid()
    metadata = await _create_asset(sid)

    from starlette.concurrency import run_in_threadpool as real_run_in_threadpool

    async def _failing_run_in_threadpool(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "unlink":
            raise OSError("simulated unlink failure")
        return await real_run_in_threadpool(func, *args, **kwargs)

    monkeypatch.setattr(session_assets, "run_in_threadpool", _failing_run_in_threadpool)

    # A failed unlink is logged, not propagated: Redis is already the
    # source of truth and is already cleared by this point, so the delete
    # still reports success -- the leftover file becomes the sweep's job.
    result = await session_assets.delete_session_asset(metadata["asset_id"], sid)
    assert result is True

    assert await session_assets.get_session_asset_metadata(metadata["asset_id"], sid) is None
    assert await redis_module.redis.get(session_assets._asset_key(sid, metadata["asset_id"])) is None


@pytest.mark.asyncio
async def test_delete_session_asset_is_idempotent():
    sid = _valid_sid()
    metadata = await _create_asset(sid)

    assert await session_assets.delete_session_asset(metadata["asset_id"], sid) is True
    assert await session_assets.delete_session_asset(metadata["asset_id"], sid) is False


@pytest.mark.asyncio
async def test_delete_all_session_assets_deletes_everything_and_noops_on_empty():
    sid = _valid_sid()
    await _create_asset(sid)
    await _create_asset(sid)

    count = await session_assets.delete_all_session_assets(sid)
    assert count == 2
    assert await session_assets.list_session_assets(sid) == []

    count_again = await session_assets.delete_all_session_assets(sid)
    assert count_again == 0
