"""Task-local Redis cache for Speaker Verification embeddings and batch results.

Redis orchestration is kept to async functions here so synchronous
model-inference code in ``service.py`` never touches Redis directly. Every
getter degrades any failure (connection error, malformed JSON, wrong schema,
invalid values, corrupted entries) to a plain miss (``None``) so a cache
problem can never break verification -- callers always fall back to normal
analysis.

Cache keys are canonical SHA-256 digests over a sorted-key JSON object of the
values that make two requests identical. Raw audio, filenames, speaker names,
absolute paths, and auth secrets are never placed in a key or a payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from redis.exceptions import RedisError

from app.core import redis as redis_module

EMBEDDING_CACHE_SCHEMA_VERSION = "embedding-cache-v1"
BATCH_RESULT_SCHEMA_VERSION = "batch-result-v4"

EMBEDDING_CACHE_TTL_SECONDS = 24 * 60 * 60
BATCH_RESULT_CACHE_TTL_SECONDS = 6 * 60 * 60


def hash_audio_files(paths: Sequence[str | Path]) -> list[str]:
    """SHA-256 hex digest of each file's raw bytes, in the given order.

    Synchronous file I/O -- callers must invoke this via a threadpool rather
    than directly on the async event loop.
    """

    digests: list[str] = []
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests.append(digest.hexdigest())
    return digests


def _canonical_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedding_cache_key(
    *,
    model_key: str,
    model_id: str,
    revision: str,
    preprocessing_version: str,
    audio_sha256: str,
) -> str:
    """Identity: preprocessing + model + revision + audio content only.

    Pair/clustering thresholds are deliberately excluded -- they do not
    affect embedding extraction.
    """

    digest = _canonical_digest(
        {
            "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
            "preprocessing_version": preprocessing_version,
            "model_key": model_key,
            "model_id": model_id,
            "revision": revision,
            "audio_sha256": audio_sha256,
        }
    )
    return f"verify:emb:{model_key}:{digest}"


def batch_result_cache_key(
    *,
    model_key: str,
    model_id: str,
    revision: str,
    preprocessing_version: str,
    pair_threshold_version: str,
    pair_threshold_value: float,
    clustering_threshold_version: str,
    clustering_distance_threshold: float,
    recordings: Sequence[tuple[str, str]],
) -> str:
    """Identity includes ordered (audio hash, label) pairs -- order-sensitive
    by design, since request order is part of the response contract."""

    digest = _canonical_digest(
        {
            "schema": BATCH_RESULT_SCHEMA_VERSION,
            "preprocessing_version": preprocessing_version,
            "model_key": model_key,
            "model_id": model_id,
            "revision": revision,
            "pair_threshold_version": pair_threshold_version,
            "pair_threshold_value": pair_threshold_value,
            "clustering_threshold_version": clustering_threshold_version,
            "clustering_distance_threshold": clustering_distance_threshold,
            "recordings": [
                {"audio_sha256": audio_sha256, "label": label}
                for audio_sha256, label in recordings
            ],
        }
    )
    return f"verify:batch:{model_key}:{digest}"


async def get_embedding(key: str, expected_dimension: int) -> list[float] | None:
    """Raw embedding values for a cache hit, or None on any miss/corruption.

    Only structural/type sanity is checked here. Numeric validation and
    L2-renormalization are owned by ``service.rehydrate_cached_embedding``,
    which already owns embedding validation for freshly extracted vectors.
    """

    try:
        raw = await redis_module.redis.get(key)
    except RedisError:
        return None
    if raw is None:
        return None

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != EMBEDDING_CACHE_SCHEMA_VERSION:
        return None

    values = payload.get("embedding")
    if not isinstance(values, list) or len(values) != expected_dimension:
        return None
    if not all(isinstance(value, (int, float)) for value in values):
        return None

    return [float(value) for value in values]


async def set_embedding(key: str, model_key: str, embedding: Sequence[float]) -> None:
    payload = {
        "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
        "model_key": model_key,
        "embedding_dimension": len(embedding),
        "embedding": list(embedding),
    }
    try:
        await redis_module.redis.set(key, json.dumps(payload), ex=EMBEDDING_CACHE_TTL_SECONDS)
    except RedisError:
        return


def _is_finite_vector(vector: object, size: int) -> bool:
    if not isinstance(vector, list) or len(vector) != size:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in vector
    )


def _is_finite_matrix(matrix: object, size: int) -> bool:
    if not isinstance(matrix, list) or len(matrix) != size:
        return False
    return all(_is_finite_vector(row, size) for row in matrix)


def _is_boolean_matrix(matrix: object, size: int) -> bool:
    if not isinstance(matrix, list) or len(matrix) != size:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return False
        if not all(isinstance(value, bool) for value in row):
            return False
    return True


def _validate_batch_result(
    result: object,
    *,
    model_key: str,
    expected_labels: Sequence[str],
    pair_threshold_value: float,
    clustering_threshold_version: str,
    clustering_distance_threshold: float,
) -> bool:
    if not isinstance(result, dict):
        return False

    recording_count = len(expected_labels)
    if result.get("model") != model_key:
        return False
    if result.get("recording_count") != recording_count:
        return False
    if result.get("labels") != list(expected_labels):
        return False
    if result.get("threshold") != pair_threshold_value:
        return False
    if result.get("clustering_threshold_version") != clustering_threshold_version:
        return False
    if result.get("clustering_distance_threshold") != clustering_distance_threshold:
        return False

    embedding_dimension = result.get("embedding_dimension")
    if not isinstance(embedding_dimension, int) or isinstance(embedding_dimension, bool):
        return False
    if embedding_dimension <= 0:
        return False

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != recording_count:
        return False
    if not all(_is_finite_vector(vector, embedding_dimension) for vector in embeddings):
        return False

    if not _is_finite_matrix(result.get("similarity_matrix"), recording_count):
        return False
    if not _is_boolean_matrix(result.get("decision_matrix"), recording_count):
        return False

    cluster_labels = result.get("cluster_labels")
    if not isinstance(cluster_labels, list) or len(cluster_labels) != recording_count:
        return False
    if not all(isinstance(label, str) for label in cluster_labels):
        return False

    if not _is_finite_vector(result.get("cluster_fit_scores"), recording_count):
        return False

    cluster_count = result.get("cluster_count")
    if cluster_count != len(set(cluster_labels)):
        return False

    summaries = result.get("cluster_summaries")
    if not isinstance(summaries, list) or len(summaries) != cluster_count:
        return False

    return True


async def get_batch_result(
    key: str,
    *,
    model_key: str,
    expected_labels: Sequence[str],
    pair_threshold_value: float,
    clustering_threshold_version: str,
    clustering_distance_threshold: float,
) -> dict[str, object] | None:
    """Validated cache hit's bare ``result`` payload, or None.

    Only ``result`` is ever returned -- the ``{schema, result}`` envelope
    never leaks into the API response, so a hit and a fresh computation are
    contract-identical to the caller.
    """

    try:
        raw = await redis_module.redis.get(key)
    except RedisError:
        return None
    if raw is None:
        return None

    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(envelope, dict) or envelope.get("schema") != BATCH_RESULT_SCHEMA_VERSION:
        return None

    result = envelope.get("result")
    if not _validate_batch_result(
        result,
        model_key=model_key,
        expected_labels=expected_labels,
        pair_threshold_value=pair_threshold_value,
        clustering_threshold_version=clustering_threshold_version,
        clustering_distance_threshold=clustering_distance_threshold,
    ):
        return None

    return result


async def set_batch_result(key: str, result: dict[str, object]) -> None:
    envelope = {"schema": BATCH_RESULT_SCHEMA_VERSION, "result": result}
    try:
        await redis_module.redis.set(key, json.dumps(envelope), ex=BATCH_RESULT_CACHE_TTL_SECONDS)
    except RedisError:
        return
