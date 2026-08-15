"""Focused tests for the Speaker Verification Redis cache (Commit 5)."""

import hashlib
import io
import json
import threading
from pathlib import Path

import pytest
import torch
from redis.exceptions import RedisError

from app.core import redis as redis_module
from app.tasks.verification import cache, clustering, service

ECAPA_SPEC = service.MODEL_SPECS["ecapa-tdnn"]
ECAPA_DIM = ECAPA_SPEC.embedding_dimension


def _embedding_key(audio_sha256: str) -> str:
    return cache.embedding_cache_key(
        model_key=ECAPA_SPEC.key,
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        audio_sha256=audio_sha256,
    )


def _batch_key(recordings: list[tuple[str, str]]) -> str:
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    return cache.batch_result_cache_key(
        model_key=ECAPA_SPEC.key,
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        pair_threshold_version=service.PAIR_THRESHOLD_VERSION,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
        recordings=recordings,
    )


def _valid_embedding_payload(dimension: int = ECAPA_DIM) -> dict:
    values = [0.0] * dimension
    values[0] = 1.0
    return {
        "schema": cache.EMBEDDING_CACHE_SCHEMA_VERSION,
        "model_key": ECAPA_SPEC.key,
        "embedding_dimension": dimension,
        "embedding": values,
    }


def _valid_batch_result(labels: list[str], clustering_spec) -> dict:
    n = len(labels)
    embeddings = [[1.0] + [0.0] * (ECAPA_DIM - 1) for _ in range(n)]
    similarity = [[1.0 if i == j else 0.9 for j in range(n)] for i in range(n)]
    decisions = [[True for _ in range(n)] for _ in range(n)]
    cluster_labels = ["cluster-1"] * n
    return {
        "model": ECAPA_SPEC.key,
        "model_label": ECAPA_SPEC.label,
        "threshold": ECAPA_SPEC.threshold,
        "recording_count": n,
        "embedding_dimension": ECAPA_DIM,
        "labels": list(labels),
        "embeddings": embeddings,
        "similarity_matrix": similarity,
        "decision_matrix": decisions,
        "clustering_distance_threshold": clustering_spec.distance_threshold,
        "clustering_threshold_version": clustering.CLUSTERING_THRESHOLD_VERSION,
        "cluster_labels": cluster_labels,
        "cluster_fit_scores": [0.0] * n,
        "cluster_count": 1,
        "cluster_summaries": [{"cluster_id": "cluster-1", "member_count": n}],
    }


# --------------------------------------------------------------------------
# Key stability / no-leak tests (pure functions, no Redis involved)
# --------------------------------------------------------------------------


def test_embedding_key_is_stable_for_identical_inputs():
    key_a = _embedding_key("a" * 64)
    key_b = _embedding_key("a" * 64)
    assert key_a == key_b
    assert key_a.startswith(f"verify:emb:{ECAPA_SPEC.key}:")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kwargs: kwargs.update(model_key="resnet34-lm"),
        lambda kwargs: kwargs.update(model_id="different/model"),
        lambda kwargs: kwargs.update(revision="different-revision"),
        lambda kwargs: kwargs.update(preprocessing_version="different-preprocess-v2"),
        lambda kwargs: kwargs.update(audio_sha256="b" * 64),
    ],
)
def test_embedding_key_changes_when_any_identity_input_changes(mutate):
    base_kwargs = dict(
        model_key=ECAPA_SPEC.key,
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        audio_sha256="a" * 64,
    )
    baseline = cache.embedding_cache_key(**base_kwargs)

    mutated_kwargs = dict(base_kwargs)
    mutate(mutated_kwargs)
    mutated = cache.embedding_cache_key(**mutated_kwargs)

    assert mutated != baseline


def test_embedding_key_excludes_thresholds_by_construction():
    # embedding_cache_key has no threshold parameters at all -- this is a
    # compile-time guarantee, exercised here by confirming the same audio +
    # model + revision always yields the same key regardless of what pair/
    # clustering thresholds are in effect elsewhere in the system.
    key_1 = _embedding_key("c" * 64)
    key_2 = _embedding_key("c" * 64)
    assert key_1 == key_2


def test_batch_result_key_is_stable_for_identical_inputs():
    recordings = [("a" * 64, "rec_a"), ("b" * 64, "rec_b")]
    assert _batch_key(recordings) == _batch_key(recordings)


def test_batch_result_key_is_order_sensitive():
    forward = [("a" * 64, "rec_a"), ("b" * 64, "rec_b")]
    reversed_order = [("b" * 64, "rec_b"), ("a" * 64, "rec_a")]
    assert _batch_key(forward) != _batch_key(reversed_order)


@pytest.mark.parametrize(
    "kwargs_override",
    [
        {"model_key": "resnet34-lm"},
        {"model_id": "different/model"},
        {"revision": "different-revision"},
        {"preprocessing_version": "different-preprocess-v2"},
        {"pair_threshold_version": "pair-threshold-v2-eer"},
        {"pair_threshold_value": 0.1},
        {"clustering_threshold_version": "clustering-v2"},
        {"clustering_distance_threshold": 0.1},
    ],
)
def test_batch_result_key_changes_when_any_identity_input_changes(kwargs_override):
    recordings = [("a" * 64, "rec_a"), ("b" * 64, "rec_b")]
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    base_kwargs = dict(
        model_key=ECAPA_SPEC.key,
        model_id=ECAPA_SPEC.model_id,
        revision=ECAPA_SPEC.revision,
        preprocessing_version=service.PREPROCESSING_VERSION,
        pair_threshold_version=service.PAIR_THRESHOLD_VERSION,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
        recordings=recordings,
    )
    baseline = cache.batch_result_cache_key(**base_kwargs)

    mutated_kwargs = dict(base_kwargs)
    mutated_kwargs.update(kwargs_override)
    mutated = cache.batch_result_cache_key(**mutated_kwargs)

    assert mutated != baseline


def test_keys_and_payloads_never_contain_secret_filenames_or_paths():
    secret = "C:/Users/DELL/secret_name_alice.wav"
    audio_sha256 = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    key = _embedding_key(audio_sha256)
    payload = _valid_embedding_payload()

    assert "secret_name_alice" not in key
    assert secret not in key
    assert "secret_name_alice" not in json.dumps(payload)


# --------------------------------------------------------------------------
# hash_audio_files
# --------------------------------------------------------------------------


def test_hash_audio_files_matches_hashlib_and_preserves_order(tmp_path):
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    path_a.write_bytes(b"AUDIO-CONTENT-A")
    path_b.write_bytes(b"AUDIO-CONTENT-B")

    digests = cache.hash_audio_files([path_a, path_b])

    assert digests == [
        hashlib.sha256(b"AUDIO-CONTENT-A").hexdigest(),
        hashlib.sha256(b"AUDIO-CONTENT-B").hexdigest(),
    ]


# --------------------------------------------------------------------------
# Embedding cache round trip / TTL / corruption
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_cache_round_trip_and_ttl():
    key = _embedding_key("d" * 64)
    embedding = [1.0] + [0.0] * (ECAPA_DIM - 1)

    assert await cache.get_embedding(key, ECAPA_DIM) is None

    await cache.set_embedding(key, ECAPA_SPEC.key, embedding)
    fetched = await cache.get_embedding(key, ECAPA_DIM)
    assert fetched == embedding

    ttl = await redis_module.redis.ttl(key)
    assert 0 < ttl <= cache.EMBEDDING_CACHE_TTL_SECONDS
    assert ttl > cache.EMBEDDING_CACHE_TTL_SECONDS - 60


@pytest.mark.asyncio
async def test_embedding_cache_wrong_dimension_is_a_miss():
    key = _embedding_key("e" * 64)
    await redis_module.redis.set(key, json.dumps(_valid_embedding_payload(dimension=8)), ex=3600)

    assert await cache.get_embedding(key, ECAPA_DIM) is None


@pytest.mark.asyncio
async def test_embedding_cache_malformed_json_is_a_miss():
    key = _embedding_key("f" * 64)
    await redis_module.redis.set(key, "not-json{{{", ex=3600)

    assert await cache.get_embedding(key, ECAPA_DIM) is None


@pytest.mark.asyncio
async def test_embedding_cache_wrong_schema_is_a_miss():
    key = _embedding_key("g" * 64)
    payload = _valid_embedding_payload()
    payload["schema"] = "embedding-cache-v0-old"
    await redis_module.redis.set(key, json.dumps(payload), ex=3600)

    assert await cache.get_embedding(key, ECAPA_DIM) is None


@pytest.mark.asyncio
async def test_embedding_cache_non_finite_values_pass_structural_check_but_fail_rehydration():
    # cache.get_embedding only checks shape/type; semantic validation
    # (finite, non-zero norm) is owned by service.rehydrate_cached_embedding.
    key = _embedding_key("h" * 64)
    values = [0.0] * ECAPA_DIM  # zero-magnitude: structurally fine, semantically invalid
    await redis_module.redis.set(
        key,
        json.dumps({**_valid_embedding_payload(), "embedding": values}),
        ex=3600,
    )

    raw = await cache.get_embedding(key, ECAPA_DIM)
    assert raw is not None  # structural check passes

    rehydrated = service.rehydrate_cached_embedding(ECAPA_SPEC.key, raw)
    assert rehydrated is None  # semantic check rejects it


# --------------------------------------------------------------------------
# rehydrate_cached_embedding
# --------------------------------------------------------------------------


def test_rehydrate_cached_embedding_normalizes_valid_values():
    values = [3.0] + [0.0] * (ECAPA_DIM - 1)
    tensor = service.rehydrate_cached_embedding(ECAPA_SPEC.key, values)
    assert tensor is not None
    assert torch.allclose(torch.linalg.vector_norm(tensor), torch.tensor(1.0), atol=1e-5)


def test_rehydrate_cached_embedding_rejects_wrong_dimension():
    assert service.rehydrate_cached_embedding(ECAPA_SPEC.key, [1.0, 2.0]) is None


def test_rehydrate_cached_embedding_rejects_non_finite():
    values = [float("nan")] + [0.0] * (ECAPA_DIM - 1)
    assert service.rehydrate_cached_embedding(ECAPA_SPEC.key, values) is None


def test_rehydrate_cached_embedding_rejects_zero_norm():
    values = [0.0] * ECAPA_DIM
    assert service.rehydrate_cached_embedding(ECAPA_SPEC.key, values) is None


# --------------------------------------------------------------------------
# Batch-result cache round trip / TTL / corruption
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_result_cache_round_trip_and_ttl():
    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    result = _valid_batch_result(labels, clustering_spec)

    lookup_kwargs = dict(
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )

    assert await cache.get_batch_result(key, **lookup_kwargs) is None

    await cache.set_batch_result(key, result)
    fetched = await cache.get_batch_result(key, **lookup_kwargs)
    assert fetched == result
    # Only the bare result is returned -- no {schema, result} envelope leaks.
    assert "schema" not in fetched

    ttl = await redis_module.redis.ttl(key)
    assert 0 < ttl <= cache.BATCH_RESULT_CACHE_TTL_SECONDS
    assert ttl > cache.BATCH_RESULT_CACHE_TTL_SECONDS - 60


@pytest.mark.asyncio
async def test_batch_result_cache_tampered_recording_count_is_a_miss():
    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    result = _valid_batch_result(labels, clustering_spec)
    result["recording_count"] = 99
    await redis_module.redis.set(
        key,
        json.dumps({"schema": cache.BATCH_RESULT_SCHEMA_VERSION, "result": result}),
        ex=3600,
    )

    fetched = await cache.get_batch_result(
        key,
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    assert fetched is None


@pytest.mark.asyncio
async def test_batch_result_cache_reordered_labels_is_a_miss():
    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    result = _valid_batch_result(labels, clustering_spec)
    result["labels"] = list(reversed(labels))
    await redis_module.redis.set(
        key,
        json.dumps({"schema": cache.BATCH_RESULT_SCHEMA_VERSION, "result": result}),
        ex=3600,
    )

    fetched = await cache.get_batch_result(
        key,
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    assert fetched is None


@pytest.mark.asyncio
async def test_batch_result_cache_non_finite_similarity_value_is_a_miss():
    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    result = _valid_batch_result(labels, clustering_spec)
    result["similarity_matrix"][0][1] = float("nan")
    await redis_module.redis.set(
        key,
        json.dumps({"schema": cache.BATCH_RESULT_SCHEMA_VERSION, "result": result}),
        ex=3600,
    )

    fetched = await cache.get_batch_result(
        key,
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    assert fetched is None


@pytest.mark.asyncio
async def test_batch_result_cache_wrong_schema_is_a_miss():
    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    result = _valid_batch_result(labels, clustering_spec)
    await redis_module.redis.set(
        key,
        json.dumps({"schema": "batch-result-v0-old", "result": result}),
        ex=3600,
    )

    fetched = await cache.get_batch_result(
        key,
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    assert fetched is None


# --------------------------------------------------------------------------
# Redis unavailable -> cache degrades to a miss, never raises
# --------------------------------------------------------------------------


class _RaisingRedis:
    async def get(self, key):
        raise RedisError("simulated connection failure")

    async def set(self, key, value, ex=None):
        raise RedisError("simulated connection failure")

    async def ttl(self, key):
        raise RedisError("simulated connection failure")


@pytest.mark.asyncio
async def test_cache_functions_survive_redis_errors(monkeypatch):
    monkeypatch.setattr(redis_module, "redis", _RaisingRedis())

    embedding_result = await cache.get_embedding(_embedding_key("i" * 64), ECAPA_DIM)
    assert embedding_result is None

    # Must not raise.
    await cache.set_embedding(_embedding_key("i" * 64), ECAPA_SPEC.key, [0.0] * ECAPA_DIM)

    labels = ["rec_a", "rec_b"]
    recordings = [("a" * 64, labels[0]), ("b" * 64, labels[1])]
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    batch_result = await cache.get_batch_result(
        _batch_key(recordings),
        model_key=ECAPA_SPEC.key,
        expected_labels=labels,
        pair_threshold_value=ECAPA_SPEC.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    assert batch_result is None

    # Must not raise.
    await cache.set_batch_result(_batch_key(recordings), _valid_batch_result(labels, clustering_spec))


# --------------------------------------------------------------------------
# Router-level integration: full hit/miss flow via /batch/upload and
# /batch/dataset, with a content-keyed fake adapter so real models are never
# loaded and cache identity is driven by audio bytes, not paths/labels.
# --------------------------------------------------------------------------


class _ContentKeyedAdapter:
    """Fake adapter returning a deterministic embedding per audio content.

    Mirrors real adapters' guarantee that `extract_embedding` returns an
    L2-normalized vector.
    """

    def __init__(self, embeddings_by_content: dict[bytes, torch.Tensor]) -> None:
        self.embeddings_by_content = embeddings_by_content
        self.calls: list[bytes] = []

    def extract_embedding(self, audio_path):
        content = Path(audio_path).read_bytes()
        self.calls.append(content)
        return self.embeddings_by_content[content]


def _make_embedding(seed: float) -> torch.Tensor:
    values = torch.zeros(ECAPA_DIM)
    values[0] = seed
    values[1] = 1.0
    return torch.nn.functional.normalize(values, p=2, dim=0)


CONTENT_A = b"AUDIO-CONTENT-A"
CONTENT_B = b"AUDIO-CONTENT-B"
CONTENT_C = b"AUDIO-CONTENT-C"
CONTENT_D = b"AUDIO-CONTENT-D"
CONTENT_E = b"AUDIO-CONTENT-E"
CONTENT_F = b"AUDIO-CONTENT-F"
CONTENT_G = b"AUDIO-CONTENT-G"


def _install_fake_adapter(monkeypatch, embeddings_by_content: dict[bytes, torch.Tensor]) -> _ContentKeyedAdapter:
    adapter = _ContentKeyedAdapter(embeddings_by_content)
    monkeypatch.setattr(service, "get_model", lambda _model_key: adapter)
    return adapter


def _upload_files(contents: list[bytes]):
    return [
        ("files", (f"f{i}.wav", io.BytesIO(content), "audio/wav"))
        for i, content in enumerate(contents)
    ]


@pytest.mark.asyncio
async def test_batch_upload_full_result_cache_hit_skips_all_inference(monkeypatch, client):
    embeddings = {CONTENT_A: _make_embedding(1.0), CONTENT_B: _make_embedding(2.0)}
    adapter = _install_fake_adapter(monkeypatch, embeddings)

    first = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )
    assert first.status_code == 200
    assert len(adapter.calls) == 2

    second = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )
    assert second.status_code == 200
    assert len(adapter.calls) == 2  # no new extraction: full batch-result hit
    assert second.json() == first.json()


@pytest.mark.asyncio
async def test_batch_upload_partial_embedding_hit_computes_only_missing(monkeypatch, client):
    embeddings = {
        CONTENT_A: _make_embedding(1.0),
        CONTENT_B: _make_embedding(2.0),
        CONTENT_C: _make_embedding(3.0),
    }
    adapter = _install_fake_adapter(monkeypatch, embeddings)

    # Warm the cache for A and B as a pair.
    warm = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )
    assert warm.status_code == 200
    assert len(adapter.calls) == 2

    # New pairing (A, C): the batch-result itself is a miss (different
    # recordings), but A's embedding should be reused from cache.
    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_C]),
    )
    assert response.status_code == 200
    assert adapter.calls.count(CONTENT_A) == 1  # not recomputed
    assert adapter.calls.count(CONTENT_C) == 1  # computed once
    assert len(adapter.calls) == 3


@pytest.mark.asyncio
async def test_batch_upload_invalid_cached_embedding_is_recomputed_in_same_request(monkeypatch, client):
    """Correction 1: an invalid cache hit must be folded into the missing
    set before the recompute batch is built, not silently skipped."""

    embeddings = {CONTENT_D: _make_embedding(4.0), CONTENT_E: _make_embedding(5.0)}
    adapter = _install_fake_adapter(monkeypatch, embeddings)

    # Pre-seed a structurally-valid-but-semantically-invalid (zero-norm)
    # embedding for D's content hash -- this passes cache.get_embedding's
    # shape check but must fail service.rehydrate_cached_embedding.
    audio_hash_d = hashlib.sha256(CONTENT_D).hexdigest()
    key_d = _embedding_key(audio_hash_d)
    await redis_module.redis.set(
        key_d,
        json.dumps({**_valid_embedding_payload(), "embedding": [0.0] * ECAPA_DIM}),
        ex=3600,
    )

    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_D, CONTENT_E]),
    )
    assert response.status_code == 200
    # Both D (invalid cache hit -> recomputed) and E (genuine miss) must have
    # been extracted in this single request.
    assert adapter.calls.count(CONTENT_D) == 1
    assert adapter.calls.count(CONTENT_E) == 1


@pytest.mark.asyncio
async def test_batch_upload_tampered_batch_result_falls_through_to_recomputation(monkeypatch, client):
    """Correction 2: a tampered/invalid cached batch result must not be
    returned -- the request must recompute instead."""

    embeddings = {CONTENT_F: _make_embedding(6.0), CONTENT_G: _make_embedding(7.0)}
    adapter = _install_fake_adapter(monkeypatch, embeddings)

    labels = ["upload-000", "upload-001"]
    recordings = [
        (hashlib.sha256(CONTENT_F).hexdigest(), labels[0]),
        (hashlib.sha256(CONTENT_G).hexdigest(), labels[1]),
    ]
    key = _batch_key(recordings)
    clustering_spec = clustering.CLUSTERING_SPECS[ECAPA_SPEC.key]
    tampered = _valid_batch_result(labels, clustering_spec)
    tampered["recording_count"] = 999  # corrupt a field validated on read
    await redis_module.redis.set(
        key,
        json.dumps({"schema": cache.BATCH_RESULT_SCHEMA_VERSION, "result": tampered}),
        ex=3600,
    )

    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_F, CONTENT_G]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recording_count"] == 2  # real computed result, not the tampered one
    assert adapter.calls.count(CONTENT_F) == 1
    assert adapter.calls.count(CONTENT_G) == 1


@pytest.mark.asyncio
async def test_batch_upload_hashes_audio_off_the_event_loop(monkeypatch, client):
    """Correction 3: audio hashing must run via the threadpool, not inline
    on the async event loop."""

    embeddings = {CONTENT_A: _make_embedding(1.0), CONTENT_B: _make_embedding(2.0)}
    _install_fake_adapter(monkeypatch, embeddings)

    original = cache.hash_audio_files
    threads_used: list[threading.Thread] = []

    def _tracking_hash(paths):
        threads_used.append(threading.current_thread())
        return original(paths)

    monkeypatch.setattr(cache, "hash_audio_files", _tracking_hash)

    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )
    assert response.status_code == 200
    assert threads_used
    assert all(thread is not threading.main_thread() for thread in threads_used)


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_uses_cache(monkeypatch, client, tmp_path):
    from app.core.settings import settings
    from app.tasks.verification import dataset

    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir()
    (dataset_dir / "id10018_01.wav").write_bytes(CONTENT_A)
    (dataset_dir / "id10324_01.wav").write_bytes(CONTENT_B)
    (dataset_dir / "metadata.csv").write_text(
        "file_name,speaker_name\nid10018_01.wav,Akshay_Kumar\nid10324_01.wav,Freida_Pinto\n"
    )
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)

    embeddings = {CONTENT_A: _make_embedding(1.0), CONTENT_B: _make_embedding(2.0)}
    adapter = _install_fake_adapter(monkeypatch, embeddings)

    ids = [recording.recording_id for recording in dataset.list_recordings()]

    first = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "ecapa-tdnn", "recording_ids": ids},
    )
    assert first.status_code == 200
    assert len(adapter.calls) == 2

    second = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "ecapa-tdnn", "recording_ids": ids},
    )
    assert second.status_code == 200
    assert len(adapter.calls) == 2  # full batch-result cache hit
    assert second.json() == first.json()


@pytest.mark.asyncio
async def test_batch_upload_response_uncached_matches_cached(monkeypatch, client):
    """A cached response and a freshly computed response must be identical."""

    embeddings = {CONTENT_A: _make_embedding(1.0), CONTENT_B: _make_embedding(2.0)}
    _install_fake_adapter(monkeypatch, embeddings)

    uncached = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )
    cached = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=_upload_files([CONTENT_A, CONTENT_B]),
    )

    assert uncached.status_code == cached.status_code == 200
    assert uncached.json() == cached.json()
