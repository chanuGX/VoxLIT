"""Speaker Verification endpoints (mounted at /tasks/verification)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from dataclasses import asdict

from . import cache, clustering
from .dataset import (
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .service import (
    PAIR_THRESHOLD_VERSION,
    PREPROCESSING_VERSION,
    SpeakerModelUnavailable,
    UnsupportedSpeakerModel,
    assemble_batch_analysis,
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



@router.post("/verify")
async def run_verification(
    model: str = Form(...),
    enrollment_files: list[UploadFile] = File(...),
    probe_file: UploadFile = File(...),
):
    """Verify one probe against a temporary profile built from 3–5 clips."""

    if not 3 <= len(enrollment_files) <= 5:
        raise HTTPException(
            status_code=422,
            detail="Upload between 3 and 5 enrolment recordings.",
        )

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-") as temp_dir:
            temp_path = Path(temp_dir)
            enrollment_paths: list[Path] = []
            for index, upload in enumerate(enrollment_files):
                suffix = _validate_upload(upload)
                path = temp_path / f"enrollment-{index}{suffix}"
                _save_upload(upload, path)
                enrollment_paths.append(path)

            probe_suffix = _validate_upload(probe_file)
            probe_path = temp_path / f"probe{probe_suffix}"
            _save_upload(probe_file, probe_path)

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
        for upload in [*enrollment_files, probe_file]:
            await upload.close()


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
async def run_batch_dataset(payload: BatchDatasetRequest):
    """Pairwise embedding/similarity analysis for 2-100 demo-dataset recordings."""

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

    try:
        audio_paths = [resolve_recording_path(recording_id) for recording_id in recording_ids]
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RecordingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

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


@router.post("/explain/temporal-occlusion")
async def run_temporal_occlusion(
    model: str = Form(...),
    enrollment_files: list[UploadFile] = File(...),
    probe_file: UploadFile = File(...),
    segment_count: int = Form(8),
):
    """Return probe-segment importance based on cosine-score changes."""

    if not 3 <= len(enrollment_files) <= 5:
        raise HTTPException(status_code=422, detail="Upload between 3 and 5 enrolment recordings.")
    if not 4 <= segment_count <= 20:
        raise HTTPException(status_code=422, detail="Choose between 4 and 20 occlusion segments.")

    try:
        get_model_spec(model)
    except UnsupportedSpeakerModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        with TemporaryDirectory(prefix="voxlit-sv-") as temp_dir:
            temp_path = Path(temp_dir)
            enrollment_paths: list[Path] = []
            for index, upload in enumerate(enrollment_files):
                suffix = _validate_upload(upload)
                path = temp_path / f"enrollment-{index}{suffix}"
                _save_upload(upload, path)
                enrollment_paths.append(path)

            probe_suffix = _validate_upload(probe_file)
            probe_path = temp_path / f"probe{probe_suffix}"
            _save_upload(probe_file, probe_path)

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
        for upload in [*enrollment_files, probe_file]:
            await upload.close()