"""Verification-safe adapter over the hardened Custom Dataset Manager.

Speaker Verification never uses the generic `custom:{session_id}:{name}`
identifier (used by Transcription/Emotion) -- it never leaves this task
package, and it is never sent to or returned by any Verification endpoint.
Instead, every custom-dataset recording exposed here gets an opaque
`crec_...` id derived deterministically from (session id, dataset name,
stored filename); nothing about the id itself is reversible, and every
resolution re-validates the caller's own canonical session and re-checks
live existence/containment through `CustomDatasetManager` -- Redis is a
performance cache for the id-to-file mapping, never the authority.

Reuses `CustomDatasetManager` for all actual storage/security logic (session
ownership, path containment, atomic writes, per-dataset locking) rather than
duplicating it -- this module only adds the `crec_` identity layer and a
narrow, session-id-free listing/summary contract on top.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from redis.exceptions import RedisError

from app.core import redis as redis_module
from app.core.session import InvalidSessionId, validate_session_id
from app.services.custom_dataset_service import CustomDatasetManager, validate_dataset_name

logger = logging.getLogger(__name__)

CREC_MAPPING_TTL_SECONDS = 24 * 60 * 60


class CustomRecordingNotFound(ValueError):
    """Raised when a `crec_...` id, or the dataset it names, is unknown,
    malformed, expired, or not owned by the caller's session."""


class CustomRecordingCollision(RuntimeError):
    """Raised when two different files in one dataset hash to the same
    `crec_...` id -- fails the whole listing closed rather than silently
    keeping one mapping."""


class CustomRecordingRegistryUnavailable(RuntimeError):
    """Raised when a Redis operation this identity scheme depends on fails.
    Never degrades to a silently-broken success or an unbounded filesystem
    scan -- the caller maps this to a 503."""


@dataclass(frozen=True, slots=True)
class CustomRecordingInfo:
    recording_id: str
    display_filename: str
    extension: str
    size_bytes: int
    duration_seconds: float | None
    dataset_name: str
    origin: str = "custom_dataset"


@dataclass(frozen=True, slots=True)
class CustomDatasetSummary:
    dataset_name: str
    total_files: int
    created_at: str | None


def _recording_id_for(sid: str, dataset_name: str, filename: str) -> str:
    digest = hashlib.sha256(f"{sid}:{dataset_name}:{filename}".encode("utf-8")).hexdigest()
    return f"crec_{digest}"


def _mapping_key(sid: str, recording_id: str) -> str:
    return f"verify:crec:map:{sid}:{recording_id}"


def list_owned_datasets_safe(sid: str) -> list[CustomDatasetSummary]:
    """Session-id-free dataset summaries for Verification's own dataset
    picker -- never `session_id`, never `formatted_name`, never a per-file
    list. An invalid/unusable session simply owns no datasets, not an error."""

    try:
        validated_sid = validate_session_id(sid)
    except InvalidSessionId:
        return []

    manager = CustomDatasetManager(validated_sid)
    datasets = manager.list_datasets()
    return [
        CustomDatasetSummary(
            dataset_name=entry["dataset_name"],
            total_files=entry.get("total_files", len(entry.get("files", []))),
            created_at=entry.get("created_at"),
        )
        for entry in datasets
    ]


async def list_custom_recordings(sid: str, dataset_name: str) -> list[CustomRecordingInfo]:
    """Builds this dataset's safe recording list, then atomically replaces
    this dataset's own Redis mappings only -- never another dataset's, so
    concurrent tabs/datasets under one session stay independently
    resolvable. Returns nothing at all if the Redis write fails, rather than
    a list whose ids were never actually registered."""

    try:
        validated_sid = validate_session_id(sid)
        safe_name = validate_dataset_name(dataset_name)
    except (InvalidSessionId, ValueError) as error:
        raise CustomRecordingNotFound(str(error)) from error

    manager = CustomDatasetManager(validated_sid)
    metadata = manager.get_dataset_metadata(safe_name)
    if metadata is None:
        raise CustomRecordingNotFound(f"Unknown custom dataset: {dataset_name}")

    seen: dict[str, str] = {}
    infos: list[CustomRecordingInfo] = []
    for file_entry in metadata["files"]:
        filename = file_entry["filename"]
        recording_id = _recording_id_for(validated_sid, safe_name, filename)
        if recording_id in seen and seen[recording_id] != filename:
            raise CustomRecordingCollision(
                f"Recording id collision within dataset {safe_name!r}: {recording_id}"
            )
        seen[recording_id] = filename
        infos.append(
            CustomRecordingInfo(
                recording_id=recording_id,
                display_filename=file_entry.get("original_filename", filename),
                extension=Path(filename).suffix.lower(),
                size_bytes=file_entry["size"],
                duration_seconds=file_entry.get("duration"),
                dataset_name=safe_name,
            )
        )

    if seen:
        try:
            pipe = redis_module.redis.pipeline(transaction=True)
            for recording_id, stored_filename in seen.items():
                payload = json.dumps({"dataset_name": safe_name, "stored_filename": stored_filename})
                pipe.set(_mapping_key(validated_sid, recording_id), payload, ex=CREC_MAPPING_TTL_SECONDS)
            await pipe.execute()
        except RedisError as error:
            raise CustomRecordingRegistryUnavailable(
                "Speaker verification storage is temporarily unavailable."
            ) from error

    return infos


async def resolve_custom_recording_path(recording_id: str, sid: str) -> Path:
    """Async single-key mapping lookup -> live filesystem recheck -> sliding
    per-key TTL refresh. Any Redis failure (lookup or TTL refresh) is a
    controlled `CustomRecordingRegistryUnavailable`, never an unhandled
    exception and never a silently-degraded success; a clean miss (Redis
    reachable, key genuinely absent) is `CustomRecordingNotFound`. There is
    no fallback that scans the filesystem or another dataset -- an id that
    isn't in this session's own current mapping is unresolvable, full stop."""

    try:
        validated_sid = validate_session_id(sid)
    except InvalidSessionId as error:
        raise CustomRecordingNotFound(str(error)) from error

    try:
        raw = await redis_module.redis.get(_mapping_key(validated_sid, recording_id))
    except RedisError as error:
        raise CustomRecordingRegistryUnavailable(
            "Speaker verification storage is temporarily unavailable."
        ) from error

    if raw is None:
        raise CustomRecordingNotFound(f"Unknown or expired custom recording id: {recording_id}")

    try:
        mapping = json.loads(raw)
        dataset_name = mapping["dataset_name"]
        stored_filename = mapping["stored_filename"]
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        raise CustomRecordingNotFound(
            f"Unknown or expired custom recording id: {recording_id}"
        ) from error

    manager = CustomDatasetManager(validated_sid)
    try:
        path = manager.resolve_file_path(dataset_name, stored_filename)
    except (FileNotFoundError, ValueError) as error:
        raise CustomRecordingNotFound(
            f"Unknown or expired custom recording id: {recording_id}"
        ) from error

    try:
        await redis_module.redis.expire(_mapping_key(validated_sid, recording_id), CREC_MAPPING_TTL_SECONDS)
    except RedisError as error:
        raise CustomRecordingRegistryUnavailable(
            "Speaker verification storage is temporarily unavailable."
        ) from error

    return path
