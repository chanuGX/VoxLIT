"""Speaker Verification endpoints (mounted at /tasks/verification)."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from dataclasses import asdict

from app.services.dataset_service import media_type_for

from app.core.session import InvalidSessionId, validate_session_id

from . import cache, clustering, custom_recordings, export, session_assets
from .audio_streaming import stream_audio_file
from .dataset import (
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .robustness import perturb_and_compare
from .service import (
    PAIR_THRESHOLD_VERSION,
    PREPROCESSING_VERSION,
    SpeakerModelUnavailable,
    UnsupportedSpeakerModel,
    assemble_batch_analysis,
    compute_saliency_map,
    extract_batch_embeddings,
    get_model_spec,
    list_models,
    project_batch_embeddings,
    rehydrate_cached_embedding,
    temporal_occlusion_saliency,
    verify_speaker,
)

router = APIRouter()

@router.get("/models")
async def available_models():
    return {"models": list_models()}


@router.get("/dataset")
async def dataset_info():
    return get_dataset_info()


@router.get("/dataset/recordings")
async def dataset_recordings():
    try:
        recordings = list_recordings()
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "dataset_id": get_dataset_info()["dataset_id"],
        "total_recordings": len(recordings),
        "recordings": [asdict(recording) for recording in recordings],
    }


@router.get("/dataset/recordings/{recording_id}")
async def dataset_recording(recording_id: str):
    try:
        recording = get_recording(recording_id)
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RecordingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return asdict(recording)


@router.get("/dataset/recordings/{recording_id}/audio")
@router.head("/dataset/recordings/{recording_id}/audio")
async def dataset_recording_audio(recording_id: str, request: Request):
    """Stream a demo recording's audio bytes by opaque id, safely.

    Resolves strictly through `resolve_recording_path`, which only ever
    compares the id against hashes of real filenames, so unknown or
    traversal-style ids simply miss and 404 without touching the filesystem
    with untrusted input. The response never carries the original filename,
    speaker name, or filesystem path — only the opaque id + extension.
    """

    try:
        audio_path = resolve_recording_path(recording_id)
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RecordingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    try:
        media_type = media_type_for(audio_path)
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error

    safe_name = f"{recording_id}{audio_path.suffix.lower()}"
    return stream_audio_file(audio_path, request, safe_name, media_type)


# ---------------------------------------------------------------------------
# Custom-dataset recordings -- Speaker Verification's own opaque `crec_...`
# identity layer over the hardened Custom Dataset Manager. Never exposes the
# generic `custom:{session_id}:{dataset_name}` identifier used by
# Transcription/Emotion; see custom_recordings.py for the full contract.
# ---------------------------------------------------------------------------


@router.get("/dataset/custom")
async def list_owned_custom_datasets(request: Request):
    """Session-id-free dataset summaries (name, file count, created-at) for
    Verification's own dataset picker/restoration -- never `session_id`,
    `formatted_name`, or a per-file list."""

    sid = _require_sid(request)
    summaries = custom_recordings.list_owned_datasets_safe(sid)
    return {"datasets": [asdict(summary) for summary in summaries]}


@router.get("/dataset/custom/{dataset_name}/recordings")
async def list_custom_dataset_recordings(dataset_name: str, request: Request):
    sid = _require_sid(request)
    try:
        recordings = await custom_recordings.list_custom_recordings(sid, dataset_name)
    except custom_recordings.CustomRecordingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except custom_recordings.CustomRecordingRegistryUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except custom_recordings.CustomRecordingCollision as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "dataset_name": dataset_name,
        "recordings": [asdict(recording) for recording in recordings],
    }


@router.get("/custom-recordings/{recording_id}/audio")
@router.head("/custom-recordings/{recording_id}/audio")
async def custom_recording_audio(recording_id: str, request: Request):
    sid = _require_sid(request)
    try:
        audio_path = await custom_recordings.resolve_custom_recording_path(recording_id, sid)
    except custom_recordings.CustomRecordingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except custom_recordings.CustomRecordingRegistryUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        media_type = media_type_for(audio_path)
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error

    safe_name = f"{recording_id}{audio_path.suffix.lower()}"
    return stream_audio_file(audio_path, request, safe_name, media_type)


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 100


class BatchDatasetRequest(BaseModel):
    model: str
    recording_ids: list[str]


SUPPORTED_REDUCTION_METHODS = {"pca", "tsne", "umap"}
SUPPORTED_PROJECTION_COMPONENTS = {2, 3}


class BatchProjectionRequest(BaseModel):
    model: str
    embeddings: list[list[float]]
    labels: list[str]
    reduction_method: str = "pca"
    n_components: int = 2


def _validate_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: {allowed}")
    return suffix


def _save_upload(file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    if destination.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Audio file is empty: {file.filename}")
    if destination.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file exceeds 50 MB: {file.filename}")


async def _resolve_audio_source(recording_id: str, sid: str) -> Path:
    """Resolve a demo `rec_...` id, a session `asset_...` id, or an owned
    custom-dataset `crec_...` id to a Path.

    Each prefix is routed to exactly one resolver and never falls through to
    another; everything else (including unknown or path-traversal-shaped
    ids) falls through to `resolve_recording_path`, which never touches the
    filesystem with untrusted input and 404s on any non-match -- unchanged
    from its existing dataset-only behavior.
    """

    if recording_id.startswith("asset_"):
        try:
            return await session_assets.resolve_session_asset_path(recording_id, sid)
        except session_assets.SessionAssetNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    if recording_id.startswith("crec_"):
        try:
            return await custom_recordings.resolve_custom_recording_path(recording_id, sid)
        except custom_recordings.CustomRecordingNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except custom_recordings.CustomRecordingRegistryUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        return resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


async def _resolve_verification_inputs(
    request: Request,
    enrollment_files: list[UploadFile],
    probe_file: UploadFile | None,
    enrollment_recording_ids: list[str],
    probe_recording_id: str | None,
    temp_dir: Path,
) -> tuple[list[Path], Path]:
    """Enrollment + probe audio for `/verify` and `/explain/temporal-occlusion`.

    A request is either fully id-based (demo/session recording ids) or fully
    upload-based (raw files) -- never a mix of both, and never neither.
    """

    using_ids = bool(enrollment_recording_ids) or bool(probe_recording_id)
    using_files = bool(enrollment_files) or probe_file is not None

    if using_ids and using_files:
        raise HTTPException(
            status_code=422,
            detail="Provide either recording ids or uploaded files, not both.",
        )

    if using_ids:
        if not enrollment_recording_ids or not 3 <= len(enrollment_recording_ids) <= 5:
            raise HTTPException(status_code=422, detail="Provide between 3 and 5 enrolment recording ids.")
        if not probe_recording_id:
            raise HTTPException(status_code=422, detail="Provide a probe recording id.")

        sid = _require_sid(request)
        try:
            validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        except session_assets.InvalidSessionId as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        enrollment_paths = [
            await _resolve_audio_source(recording_id, validated_sid)
            for recording_id in enrollment_recording_ids
        ]
        probe_path = await _resolve_audio_source(probe_recording_id, validated_sid)
        return enrollment_paths, probe_path

    if not enrollment_files or probe_file is None:
        raise HTTPException(
            status_code=422,
            detail="Upload enrolment files and a probe file, or provide recording ids.",
        )
    if not 3 <= len(enrollment_files) <= 5:
        raise HTTPException(status_code=422, detail="Upload between 3 and 5 enrolment recordings.")

    enrollment_paths = []
    for index, upload in enumerate(enrollment_files):
        suffix = _validate_upload(upload)
        path = temp_dir / f"enrollment-{index}{suffix}"
        _save_upload(upload, path)
        enrollment_paths.append(path)

    probe_suffix = _validate_upload(probe_file)
    probe_path = temp_dir / f"probe{probe_suffix}"
    _save_upload(probe_file, probe_path)
    return enrollment_paths, probe_path


@router.post("/verify")
async def run_verification(
    request: Request,
    model: str = Form(...),
    enrollment_files: list[UploadFile] = File(default_factory=list),
    probe_file: UploadFile | None = File(None),
    enrollment_recording_ids: list[str] = Form(default_factory=list),
    probe_recording_id: str | None = Form(None),
):
    """Verify one probe against a temporary profile built from 3–5 clips,
    either uploaded directly or referenced by demo/session recording id."""

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-") as temp_dir:
            temp_path = Path(temp_dir)
            enrollment_paths, probe_path = await _resolve_verification_inputs(
                request,
                enrollment_files,
                probe_file,
                enrollment_recording_ids,
                probe_recording_id,
                temp_path,
            )

            return await run_in_threadpool(
                verify_speaker,
                model,
                enrollment_paths,
                probe_path,
            )
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    finally:
        if enrollment_files:
            for upload in enrollment_files:
                await upload.close()
        if probe_file is not None:
            await probe_file.close()


async def _cached_batch_analysis(
    model_key: str,
    audio_paths: list[Path],
    labels: list[str],
) -> dict[str, object]:
    """Batch analysis with a task-local Redis cache in front of it.

    Checks a complete cached result first; on a miss, reuses cached
    per-recording embeddings and computes only the missing ones. Any Redis
    failure is already absorbed inside `cache.py`, so this always falls
    through to a normal (uncached) analysis.
    """

    spec = get_model_spec(model_key)
    clustering_spec = clustering.CLUSTERING_SPECS[model_key]

    audio_hashes = await run_in_threadpool(cache.hash_audio_files, audio_paths)
    recordings = list(zip(audio_hashes, labels))

    result_key = cache.batch_result_cache_key(
        model_key=model_key,
        model_id=spec.model_id,
        revision=spec.revision,
        preprocessing_version=PREPROCESSING_VERSION,
        pair_threshold_version=PAIR_THRESHOLD_VERSION,
        pair_threshold_value=spec.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
        recordings=recordings,
    )

    cached_result = await cache.get_batch_result(
        result_key,
        model_key=model_key,
        expected_labels=labels,
        pair_threshold_value=spec.threshold,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        clustering_distance_threshold=clustering_spec.distance_threshold,
    )
    if cached_result is not None:
        return cached_result

    embedding_keys = [
        cache.embedding_cache_key(
            model_key=model_key,
            model_id=spec.model_id,
            revision=spec.revision,
            preprocessing_version=PREPROCESSING_VERSION,
            audio_sha256=audio_hash,
        )
        for audio_hash in audio_hashes
    ]
    cached_raw = await asyncio.gather(
        *(cache.get_embedding(key, spec.embedding_dimension) for key in embedding_keys)
    )

    embeddings: list[torch.Tensor | None] = [None] * len(audio_paths)
    missing_indices: list[int] = []
    for index, raw_values in enumerate(cached_raw):
        tensor = rehydrate_cached_embedding(model_key, raw_values) if raw_values is not None else None
        embeddings[index] = tensor
        if tensor is None:
            missing_indices.append(index)

    if missing_indices:
        missing_paths = [audio_paths[index] for index in missing_indices]
        computed = await run_in_threadpool(extract_batch_embeddings, model_key, missing_paths)
        for position, index in enumerate(missing_indices):
            embeddings[index] = computed[position]

    merged_embeddings = torch.stack(embeddings)  # type: ignore[arg-type]

    result = await run_in_threadpool(
        assemble_batch_analysis, model_key, merged_embeddings, labels
    )

    await cache.set_batch_result(result_key, result)
    for index in missing_indices:
        await cache.set_embedding(embedding_keys[index], model_key, result["embeddings"][index])

    return result


@router.post("/batch/upload")
async def run_batch_upload(
    model: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Pairwise embedding/similarity analysis for 2-100 uploaded recordings."""

    if not MIN_BATCH_SIZE <= len(files) <= MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Upload between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE} recordings.",
        )

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-batch-") as temp_dir:
            temp_path = Path(temp_dir)
            audio_paths: list[Path] = []
            labels: list[str] = []
            for index, upload in enumerate(files):
                suffix = _validate_upload(upload)
                label = f"upload-{index:03d}"
                path = temp_path / f"{label}{suffix}"
                _save_upload(upload, path)
                audio_paths.append(path)
                labels.append(label)

            return await _cached_batch_analysis(model, audio_paths, labels)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    finally:
        for upload in files:
            await upload.close()


@router.post("/batch/dataset")
async def run_batch_dataset(payload: BatchDatasetRequest, request: Request):
    """Pairwise embedding/similarity analysis for 2-100 recordings, by demo
    (`rec_...`) or session-asset (`asset_...`) id, in any mix."""

    recording_ids = payload.recording_ids
    if not MIN_BATCH_SIZE <= len(recording_ids) <= MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Provide between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE} recording ids.",
        )
    if len(recording_ids) != len(set(recording_ids)):
        raise HTTPException(status_code=422, detail="Duplicate recording ids are not allowed.")

    try:
        get_model_spec(payload.model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    sid = _require_sid(request)
    audio_paths = [await _resolve_audio_source(recording_id, sid) for recording_id in recording_ids]

    try:
        return await _cached_batch_analysis(payload.model, audio_paths, recording_ids)
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error


@router.post("/batch/project")
async def run_batch_projection(payload: BatchProjectionRequest):
    """Visualization-only PCA/t-SNE/UMAP projection of precomputed batch embeddings.

    Never used for clustering — clustering runs on the original embeddings
    in /batch/upload and /batch/dataset. This endpoint exists because the
    legacy /inferences/embeddings route re-runs Whisper/Wav2Vec2 inference
    internally and cannot accept precomputed vectors or ecapa-tdnn/resnet34-lm
    model keys.
    """

    if not MIN_BATCH_SIZE <= len(payload.embeddings) <= MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Provide between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE} embeddings.",
        )
    if len(payload.embeddings) != len(payload.labels):
        raise HTTPException(status_code=422, detail="Embeddings and labels must have the same length.")
    if payload.reduction_method not in SUPPORTED_REDUCTION_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_REDUCTION_METHODS))
        raise HTTPException(status_code=400, detail=f"reduction_method must be one of: {allowed}.")
    if payload.n_components not in SUPPORTED_PROJECTION_COMPONENTS:
        raise HTTPException(status_code=422, detail="n_components must be 2 or 3.")

    try:
        get_model_spec(payload.model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        coordinates, effective_components, method_used = await run_in_threadpool(
            project_batch_embeddings,
            payload.embeddings,
            payload.model,
            payload.reduction_method,
            payload.n_components,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "labels": payload.labels,
        "model": payload.model,
        "reduction_method": payload.reduction_method,
        "reduction_method_used": method_used,
        "n_components": payload.n_components,
        "effective_components": effective_components,
        "coordinates": coordinates.tolist(),
    }


@router.post("/batch/export")
async def export_batch_analysis(payload: export.BatchExportRequest):
    """Serialize an already-computed batch analysis result (SV-FR-36) into a
    downloadable CSV. Pure serialization -- no model inference, no
    recomputed embeddings/similarity/clustering. Every metadata value in the
    CSV (model identity, both thresholds, all version strings) is derived
    server-side from the validated `model` key alone; the client never
    supplies a threshold or version value (MQ-10: a tampered payload must
    not be able to misrepresent the evidence-vs-threshold framing).

    Deliberately returns 422 for every validation failure on this endpoint,
    including an unknown model key -- unlike the 400 this router's other
    handlers use for the same `UnsupportedSpeakerModel` failure. This is
    intentional for this endpoint only; do not "fix" it to 400 for
    consistency with the rest of the router.
    """

    try:
        spec = get_model_spec(payload.model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        export.validate_export_payload(payload)
    except export.ExportValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    clustering_spec = clustering.CLUSTERING_SPECS[payload.model]
    generated_at = datetime.now(timezone.utc)
    csv_text = export.build_export_csv(
        payload,
        spec,
        clustering_spec,
        pair_threshold_version=PAIR_THRESHOLD_VERSION,
        clustering_threshold_version=clustering.CLUSTERING_THRESHOLD_VERSION,
        preprocessing_version=PREPROCESSING_VERSION,
        generated_at=generated_at,
    )

    filename = (
        f"voxlit-cluster-export-{payload.model}-"
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/explain/temporal-occlusion")
async def run_temporal_occlusion(
    request: Request,
    model: str = Form(...),
    enrollment_files: list[UploadFile] = File(default_factory=list),
    probe_file: UploadFile | None = File(None),
    enrollment_recording_ids: list[str] = Form(default_factory=list),
    probe_recording_id: str | None = Form(None),
    segment_count: int = Form(8),
):
    """Return probe-segment importance based on cosine-score changes, using
    either uploaded files or demo/session recording ids."""

    if not 4 <= segment_count <= 20:
        raise HTTPException(status_code=422, detail="Choose between 4 and 20 occlusion segments.")

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-") as temp_dir:
            temp_path = Path(temp_dir)
            enrollment_paths, probe_path = await _resolve_verification_inputs(
                request,
                enrollment_files,
                probe_file,
                enrollment_recording_ids,
                probe_recording_id,
                temp_path,
            )

            return await run_in_threadpool(
                temporal_occlusion_saliency,
                model,
                enrollment_paths,
                probe_path,
                segment_count,
            )
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    finally:
        if enrollment_files:
            for upload in enrollment_files:
                await upload.close()
        if probe_file is not None:
            await probe_file.close()


MAX_CLUSTER_REFERENCE_COUNT = 99  # batch cap (100) minus the target itself


async def _resolve_cluster_saliency_inputs(
    request: Request,
    reference_recording_ids: list[str],
    target_recording_id: str | None,
) -> tuple[list[Path], Path]:
    """1-99 unique reference ids + one target id, none overlapping.

    Cluster membership is never known to the backend -- the frontend is the
    only place a batch result (and therefore a cluster assignment) exists,
    so `reference_recording_ids` is trusted caller input, not something this
    endpoint derives or validates against any stored clustering (there is
    none; clustering.py is never imported here). What IS enforced: a sane
    count bound (matching the batch endpoints' own 2-100 cap, minus the
    excluded target), no duplicate ids (a duplicate would silently
    double-weight that recording in the centroid mean), and the target never
    appearing in its own reference set. Deliberately separate from
    `_resolve_verification_inputs`, whose 3-5 bound does not apply here."""

    if not reference_recording_ids:
        raise HTTPException(status_code=422, detail="Provide at least one reference recording id.")
    if len(reference_recording_ids) > MAX_CLUSTER_REFERENCE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Provide at most {MAX_CLUSTER_REFERENCE_COUNT} reference recording ids.",
        )
    if len(reference_recording_ids) != len(set(reference_recording_ids)):
        raise HTTPException(status_code=422, detail="Duplicate reference recording ids are not allowed.")
    if not target_recording_id:
        raise HTTPException(status_code=422, detail="Provide a target recording id.")
    if target_recording_id in reference_recording_ids:
        raise HTTPException(status_code=422, detail="Target recording must not also be a reference recording.")

    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    reference_paths = [
        await _resolve_audio_source(recording_id, validated_sid)
        for recording_id in reference_recording_ids
    ]
    target_path = await _resolve_audio_source(target_recording_id, validated_sid)
    return reference_paths, target_path


@router.post("/explain/saliency")
async def run_saliency_map(
    request: Request,
    model: str = Form(...),
    reference_type: str = Form(...),
    segment_count: int = Form(8),
    enrollment_files: list[UploadFile] = File(default_factory=list),
    probe_file: UploadFile | None = File(None),
    enrollment_recording_ids: list[str] = Form(default_factory=list),
    probe_recording_id: str | None = Form(None),
    reference_recording_ids: list[str] = Form(default_factory=list),
    target_recording_id: str | None = Form(None),
    cluster_id: str | None = Form(None),
):
    """Generalized saliency map, generalizing `/explain/temporal-occlusion`
    with a `reference_type`:

    - "cluster": >=1 reference recording ids (no upload variant -- cluster
      members always come from an already-run batch, which is always
      id-based) plus a target recording id. Enrollment/probe fields are
      rejected outright if present.
    - "enrollment": identical to `/verify`/the old occlusion endpoint --
      3-5 enrollment recordings (ids or uploaded files) plus a probe.
      Cluster-only fields are rejected outright if present.

    `/explain/temporal-occlusion` is untouched and keeps serving the raw-file
    -upload Pair Verification flow unchanged.
    """

    if reference_type not in ("cluster", "enrollment"):
        raise HTTPException(status_code=422, detail="reference_type must be 'cluster' or 'enrollment'.")
    if not 4 <= segment_count <= 20:
        raise HTTPException(status_code=422, detail="Choose between 4 and 20 occlusion segments.")

    # Strict mutual exclusivity -- fields belonging to the other mode are
    # rejected outright, never silently ignored.
    if reference_type == "cluster":
        if enrollment_files or probe_file is not None or enrollment_recording_ids or probe_recording_id:
            raise HTTPException(
                status_code=422,
                detail="reference_type=cluster does not accept enrollment/probe fields.",
            )
    else:
        if reference_recording_ids or target_recording_id or cluster_id:
            raise HTTPException(
                status_code=422,
                detail="reference_type=enrollment does not accept cluster reference/target/cluster_id fields.",
            )

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-saliency-") as temp_dir:
            temp_path = Path(temp_dir)
            if reference_type == "cluster":
                reference_paths, target_path = await _resolve_cluster_saliency_inputs(
                    request, reference_recording_ids, target_recording_id
                )
                response_cluster_id, response_target_id = cluster_id, target_recording_id
            else:
                reference_paths, target_path = await _resolve_verification_inputs(
                    request,
                    enrollment_files,
                    probe_file,
                    enrollment_recording_ids,
                    probe_recording_id,
                    temp_path,
                )
                response_cluster_id, response_target_id = None, probe_recording_id

            return await run_in_threadpool(
                compute_saliency_map,
                model,
                reference_paths,
                target_path,
                reference_type=reference_type,
                cluster_id=response_cluster_id,
                target_recording_id=response_target_id,
                segment_count=segment_count,
            )
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    finally:
        if enrollment_files:
            for upload in enrollment_files:
                await upload.close()
        if probe_file is not None:
            await probe_file.close()


class PerturbationSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class PerturbationRequest(BaseModel):
    model: str
    recording_id: str
    perturbation: PerturbationSpec


@router.post("/perturbation")
async def run_perturbation_comparison(payload: PerturbationRequest, request: Request):
    """Apply one perturbation to one selected recording (demo or session
    asset), save the perturbed clip as a new session asset, and compare its
    embedding with the original's using one loaded model adapter."""

    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        spec = get_model_spec(payload.model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    source_path = await _resolve_audio_source(payload.recording_id, validated_sid)

    audio_hash = (await run_in_threadpool(cache.hash_audio_files, [source_path]))[0]
    embedding_key = cache.embedding_cache_key(
        model_key=payload.model,
        model_id=spec.model_id,
        revision=spec.revision,
        preprocessing_version=PREPROCESSING_VERSION,
        audio_sha256=audio_hash,
    )
    cached_raw = await cache.get_embedding(embedding_key, spec.embedding_dimension)
    cached_embedding = (
        rehydrate_cached_embedding(payload.model, cached_raw) if cached_raw is not None else None
    )

    temp_path = await run_in_threadpool(session_assets.begin_asset_write, validated_sid, ".wav")
    try:
        result = await run_in_threadpool(
            perturb_and_compare,
            payload.model,
            source_path,
            payload.perturbation.type,
            payload.perturbation.params,
            temp_path,
            cached_original_embedding=cached_embedding,
        )
    except (ValueError, RuntimeError) as error:
        await run_in_threadpool(temp_path.unlink, missing_ok=True)
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    except Exception:
        # Any other exception (e.g. a bad call signature, an adapter bug) must
        # still clean up the allocated temp path before propagating -- not
        # just the two expected error types above.
        await run_in_threadpool(temp_path.unlink, missing_ok=True)
        raise

    if cached_embedding is None:
        await cache.set_embedding(embedding_key, payload.model, result.pop("original_embedding"))
    else:
        result.pop("original_embedding", None)
    result.pop("perturbed_embedding", None)

    try:
        asset_metadata = await session_assets.promote_asset(
            temp_path,
            validated_sid,
            origin="perturbation",
            extension=".wav",
            source_asset_id=payload.recording_id,
            model=payload.model,
            perturbation=result["perturbation"],
        )
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error

    return {**result, "source_recording_id": payload.recording_id, "session_asset": asset_metadata}


# ---------------------------------------------------------------------------
# Session asset registry -- secure, session-scoped audio recordings that are
# neither the permanent demo dataset nor the flat, unscoped legacy `uploads/`
# directory. Also stores the perturbed clip a `/perturbation` comparison
# produces (`origin="perturbation"`).
# ---------------------------------------------------------------------------


def _require_sid(request: Request) -> str:
    sid = getattr(request.state, "sid", None)
    if not sid:
        raise HTTPException(status_code=400, detail="No session id found.")
    return sid


@router.post("/session-assets/upload")
async def upload_session_asset(request: Request, file: UploadFile = File(...)):
    """Registers a fresh verification-task upload as a session-owned asset.

    This is the one secure ingestion point for non-demo audio in this task
    package -- unlike the legacy `/upload` route, the resulting id is bound
    to the caller's own session and cannot be resolved by any other session.
    """

    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    suffix = _validate_upload(file)
    temp_path = await run_in_threadpool(session_assets.begin_asset_write, validated_sid, suffix)
    try:
        await run_in_threadpool(_save_upload, file, temp_path)
        return await session_assets.promote_asset(
            temp_path, validated_sid, origin="upload", extension=suffix
        )
    except HTTPException:
        await run_in_threadpool(temp_path.unlink, missing_ok=True)
        raise
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error
    finally:
        await file.close()


@router.get("/session-assets")
async def list_session_assets_route(request: Request):
    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        return {"assets": await session_assets.list_session_assets(validated_sid)}
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error


@router.get("/session-assets/{asset_id}")
async def get_session_asset_route(asset_id: str, request: Request):
    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        metadata = await session_assets.get_session_asset_metadata(asset_id, validated_sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired asset id: {asset_id}")
    return metadata


@router.get("/session-assets/{asset_id}/audio")
@router.head("/session-assets/{asset_id}/audio")
async def session_asset_audio(asset_id: str, request: Request):
    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        path = await session_assets.resolve_session_asset_path(asset_id, validated_sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except session_assets.SessionAssetNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error

    try:
        media_type = media_type_for(path)
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error

    safe_name = f"{asset_id}{path.suffix.lower()}"
    return stream_audio_file(path, request, safe_name, media_type)


@router.delete("/session-assets/{asset_id}")
async def delete_session_asset_route(asset_id: str, request: Request):
    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        deleted = await session_assets.delete_session_asset(asset_id, validated_sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error
    return {"deleted": deleted}


@router.post("/session-assets/cleanup")
async def cleanup_session_assets_route(request: Request):
    sid = _require_sid(request)
    try:
        validated_sid = session_assets.validate_and_canonicalize_sid(sid)
        count = await session_assets.delete_all_session_assets(validated_sid)
    except session_assets.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="Speaker verification storage is temporarily unavailable."
        ) from error
    return {"deleted_count": count}