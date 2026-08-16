"""Secure, session-scoped audio asset registry for Speaker Verification.

Covers two origins under one registry: a freshly uploaded recording
(`origin="upload"`) and a perturbed clip produced by a later feature
(`origin="perturbation"`) both need identical security/lifecycle handling --
opaque session-owned ids, no stored filesystem path, atomic creation, sliding
TTL, and real disk cleanup -- so they are unified here rather than treated as
two ad hoc mechanisms.

Storage layout: `{BACKEND_DIR}/uploads/verification_sessions/{sid}/{asset_id}{ext}`,
structurally separate from both the flat legacy `uploads/` directory (used by
the generic, unscoped `/upload` route) and the permanent demo dataset
directory. Redis (`verify:asset:{sid}:{asset_id}` / `verify:asset:index:{sid}`)
is the source of truth for existence and ownership; it never stores a
filesystem path, only safe metadata the path is deterministically derived
from.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from pathlib import Path
from typing import Any

import torchaudio
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from app.core import redis as redis_module
from app.core.settings import BACKEND_DIR

logger = logging.getLogger(__name__)

SESSION_ASSET_SCHEMA_VERSION = "session-asset-v1"
ASSET_TTL_SECONDS = 24 * 60 * 60
SWEEP_INTERVAL_SECONDS = 15 * 60
SWEEP_LOCK_MS = int(SWEEP_INTERVAL_SECONDS * 1000 * 0.9)

_SID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SWEEP_LOCK_KEY = "verify:sweep:lock"


class InvalidSessionId(Exception):
    """Raised when a session id fails the exact-format check."""


class SessionAssetNotFound(Exception):
    """Raised when an asset id is unknown, expired, or not owned by the caller's session."""


def _storage_root() -> Path:
    return BACKEND_DIR / "uploads" / "verification_sessions"


def validate_and_canonicalize_sid(sid: str | None) -> str:
    """Full-match only against the exact 32-lowercase-hex-char shape
    `ensure_session()` actually mints (`uuid.uuid4().hex`) -- rejects
    dashed/braced/urn-prefixed/mixed-case forms that `uuid.UUID(hex=sid)`
    alone would tolerate via its internal stripping/normalization. A value
    that doesn't match exactly is rejected outright, never re-cased or
    otherwise "fixed up" -- two differently-formatted strings are never
    treated as the same session."""

    if not sid or not _SID_PATTERN.fullmatch(sid):
        raise InvalidSessionId("Session id is missing or malformed.")
    return sid


def _session_dir(validated_sid: str) -> Path:
    return _storage_root() / validated_sid


def _asset_path(validated_sid: str, asset_id: str, extension: str) -> Path:
    return _session_dir(validated_sid) / f"{asset_id}{extension}"


def _asset_key(sid: str, asset_id: str) -> str:
    return f"verify:asset:{sid}:{asset_id}"


def _index_key(sid: str) -> str:
    return f"verify:asset:index:{sid}"


def _safe_metadata(asset_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Public-facing shape -- never includes a filesystem path."""

    return {
        "asset_id": asset_id,
        "origin": record["origin"],
        "display_filename": record["display_filename"],
        "extension": record["extension"],
        "size_bytes": record["size_bytes"],
        "duration_seconds": record["duration_seconds"],
        "sample_rate": record["sample_rate"],
        "source_asset_id": record.get("source_asset_id"),
        "model": record.get("model"),
        "perturbation": record.get("perturbation"),
        "created_at": record["created_at"],
    }


# ---------------------------------------------------------------------------
# Temp-file allocation -- sync, blocking (mkdir); callers thread-pool it.
# ---------------------------------------------------------------------------


def begin_asset_write(sid: str, extension: str) -> Path:
    """Validates sid, ensures the session's storage directory exists, and
    returns a fresh temp path *inside that same directory* (so promotion is
    a same-filesystem atomic rename, not a cross-filesystem copy). Ownership:
    this function only allocates the path -- the caller must delete it if
    anything fails before `promote_asset()` is reached; `promote_asset()`
    never deletes a temp path it wasn't given to rename."""

    validated_sid = validate_and_canonicalize_sid(sid)
    session_dir = _session_dir(validated_sid)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / f".tmp-{secrets.token_hex(8)}{extension}"


# ---------------------------------------------------------------------------
# Atomic promotion: rename + inspect (threadpooled) then register (awaited).
# ---------------------------------------------------------------------------


def _finalize_asset_file(
    temp_path: Path, final_path: Path, asset_id: str, extension: str, origin: str
) -> dict[str, Any]:
    """Sync, blocking (rename + audio inspection) -- called via
    `run_in_threadpool`. No Redis access. Renames `temp_path` to
    `final_path` (atomic, same filesystem), then derives safe metadata
    purely from the file itself via `torchaudio.info()`/`stat()` -- never
    from a caller-supplied value."""

    temp_path.replace(final_path)
    info = torchaudio.info(str(final_path))
    display_filename = (
        f"perturbed_{asset_id}.wav" if origin == "perturbation" else f"{asset_id}{extension}"
    )
    return {
        "schema": SESSION_ASSET_SCHEMA_VERSION,
        "origin": origin,
        "extension": extension,
        "display_filename": display_filename,
        "size_bytes": final_path.stat().st_size,
        "duration_seconds": info.num_frames / info.sample_rate,
        "sample_rate": info.sample_rate,
        "created_at": time.time(),
    }


def _cleanup_either(temp_path: Path, final_path: Path) -> None:
    """Exactly one of these ever exists past a given point, since
    `Path.replace` is atomic -- unlink both, `missing_ok`, so the caller
    never has to know which one it was."""

    temp_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)


async def promote_asset(
    temp_path: Path,
    sid: str,
    *,
    origin: str,
    extension: str,
    source_asset_id: str | None = None,
    model: str | None = None,
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Async. Blocking rename+inspection runs in a threadpool; Redis
    registration is awaited directly here. On ANY exception after entry --
    a `torchaudio`/`stat` failure, a JSON serialization failure, or a Redis
    pipeline failure, not just `RedisError` -- whichever of
    temp_path/final_path currently exists is deleted, and the ORIGINAL
    exception is re-raised unchanged (never swallowed or re-wrapped), so the
    caller can still distinguish a content problem from an infra problem.
    The record SET, index SADD, and both EXPIREs execute inside one
    MULTI/EXEC transaction, so they can never partially apply: a mid-write
    failure can never leave an asset record without its index entry, an
    index entry without its record, or a Redis record pointing at a file
    that was never actually renamed into place."""

    validated_sid = validate_and_canonicalize_sid(sid)
    asset_id = f"asset_{secrets.token_hex(16)}"
    final_path = _asset_path(validated_sid, asset_id, extension)
    try:
        record = await run_in_threadpool(
            _finalize_asset_file, temp_path, final_path, asset_id, extension, origin
        )
        record["source_asset_id"] = source_asset_id
        record["model"] = model
        record["perturbation"] = perturbation
        payload = json.dumps(record)
        pipe = redis_module.redis.pipeline(transaction=True)
        pipe.set(_asset_key(validated_sid, asset_id), payload, ex=ASSET_TTL_SECONDS)
        pipe.sadd(_index_key(validated_sid), asset_id)
        pipe.expire(_index_key(validated_sid), ASSET_TTL_SECONDS)
        await pipe.execute()
    except Exception:
        await run_in_threadpool(_cleanup_either, temp_path, final_path)
        raise
    return _safe_metadata(asset_id, record)


# ---------------------------------------------------------------------------
# Resolve / read -- async, Redis awaited directly; every hit refreshes both
# the asset's own TTL and its session index's TTL, together, so they can
# never drift apart.
# ---------------------------------------------------------------------------


async def _get_asset_record(sid: str, asset_id: str) -> dict[str, Any] | None:
    """`RedisError` propagates naturally (not caught here) -- callers/the
    router classify it as an infra failure. A malformed/absent/
    schema-mismatched payload is a plain miss (`None`), which is a
    legitimate "doesn't exist" state, not an infra failure."""

    raw = await redis_module.redis.get(_asset_key(sid, asset_id))
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SESSION_ASSET_SCHEMA_VERSION:
        return None
    return record


async def _touch(sid: str, asset_id: str) -> None:
    """The only place either TTL is ever refreshed -- always both keys,
    together, in one transaction, so they cannot drift apart. Without this,
    an asset could stay resolvable while its session's index entry quietly
    expired, making it vanish from a restored table after a page refresh
    even though it was still individually resolvable."""

    pipe = redis_module.redis.pipeline(transaction=True)
    pipe.expire(_asset_key(sid, asset_id), ASSET_TTL_SECONDS)
    pipe.expire(_index_key(sid), ASSET_TTL_SECONDS)
    await pipe.execute()


async def resolve_session_asset_path(asset_id: str, sid: str) -> Path:
    validated_sid = validate_and_canonicalize_sid(sid)
    record = await _get_asset_record(validated_sid, asset_id)
    if record is None:
        raise SessionAssetNotFound(f"Unknown or expired asset id: {asset_id}")
    await _touch(validated_sid, asset_id)
    return _asset_path(validated_sid, asset_id, record["extension"])


async def get_session_asset_metadata(asset_id: str, sid: str) -> dict[str, Any] | None:
    validated_sid = validate_and_canonicalize_sid(sid)
    record = await _get_asset_record(validated_sid, asset_id)
    if record is None:
        return None
    await _touch(validated_sid, asset_id)
    return _safe_metadata(asset_id, record)


async def list_session_assets(sid: str) -> list[dict[str, Any]]:
    validated_sid = validate_and_canonicalize_sid(sid)
    asset_ids = await redis_module.redis.smembers(_index_key(validated_sid))
    results: list[dict[str, Any]] = []
    stale: list[str] = []
    for asset_id in asset_ids:
        record = await _get_asset_record(validated_sid, asset_id)
        if record is None:
            stale.append(asset_id)
            continue
        await _touch(validated_sid, asset_id)
        results.append(_safe_metadata(asset_id, record))
    if stale:
        await redis_module.redis.srem(_index_key(validated_sid), *stale)
    results.sort(key=lambda item: item["created_at"])
    return results


# ---------------------------------------------------------------------------
# Delete -- Redis state removed first, so a failed/crashed file-unlink can
# only ever leave an orphaned file (which the sweep already handles), never
# a Redis record pointing at a deleted file.
# ---------------------------------------------------------------------------


async def delete_session_asset(asset_id: str, sid: str) -> bool:
    validated_sid = validate_and_canonicalize_sid(sid)
    record = await _get_asset_record(validated_sid, asset_id)
    if record is None:
        return False  # already gone: idempotent success, not an error
    pipe = redis_module.redis.pipeline(transaction=True)
    pipe.delete(_asset_key(validated_sid, asset_id))
    pipe.srem(_index_key(validated_sid), asset_id)
    await pipe.execute()
    # Redis state is already gone -- the asset is unresolvable from here on
    # regardless of what happens next. A failed unlink is logged, not
    # propagated: the caller's "deleted" contract is about resolvability,
    # and any file left behind becomes the periodic sweep's problem.
    path = _asset_path(validated_sid, asset_id, record["extension"])
    try:
        await run_in_threadpool(path.unlink, missing_ok=True)
    except OSError:
        logger.exception("failed to delete session asset file; the sweep will retry")
    return True


async def delete_all_session_assets(sid: str) -> int:
    validated_sid = validate_and_canonicalize_sid(sid)
    asset_ids = await redis_module.redis.smembers(_index_key(validated_sid))
    count = 0
    for asset_id in asset_ids:
        if await delete_session_asset(asset_id, validated_sid):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Periodic sweep -- the real fallback, since Redis key expiry alone never
# deletes a disk file. Resilient to transient Redis/network failures and
# guarded against redundant concurrent execution across future workers.
# ---------------------------------------------------------------------------


def _scan_session_asset_files() -> list[tuple[str, str, Path]]:
    """Sync, blocking directory walk -- called via `run_in_threadpool`."""

    root = _storage_root()
    if not root.is_dir():
        return []
    entries: list[tuple[str, str, Path]] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir() or not _SID_PATTERN.fullmatch(session_dir.name):
            continue
        for file_path in session_dir.iterdir():
            if not file_path.is_file() or file_path.name.startswith(".tmp-"):
                continue
            entries.append((session_dir.name, file_path.stem, file_path))
    return entries


async def sweep_expired_session_assets_once() -> int:
    """Walks the storage root; for each discovered `{sid}/{asset_id}{ext}`
    file, checks whether its Redis record still exists and deletes the file
    (and its stale index entry) if not. Propagates any exception --
    `_sweep_tick` is the resilience boundary, not this function, so a caller
    can tell "a sweep ran" apart from "a sweep was skipped due to an
    error"."""

    entries = await run_in_threadpool(_scan_session_asset_files)
    deleted = 0
    for sid, asset_id, path in entries:
        if not await redis_module.redis.exists(_asset_key(sid, asset_id)):
            await run_in_threadpool(path.unlink, missing_ok=True)
            await redis_module.redis.srem(_index_key(sid), asset_id)
            deleted += 1
    return deleted


async def _sweep_tick() -> None:
    """Directly unit-testable. Acquires a short Redis lock so only one
    worker actually sweeps per interval (defensive for future multi-worker
    deployment -- no such configuration exists today). Catches ANY
    exception -- Redis outage, network error, unexpected bug -- logs it, and
    returns normally, so the loop's next sleep still happens and the next
    interval retries."""

    try:
        acquired = await redis_module.redis.set(_SWEEP_LOCK_KEY, "1", nx=True, px=SWEEP_LOCK_MS)
        if not acquired:
            return
        await sweep_expired_session_assets_once()
    except Exception:
        logger.exception("session asset sweep tick failed; retrying next interval")


async def start_sweep_loop() -> None:
    while True:
        await _sweep_tick()
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
