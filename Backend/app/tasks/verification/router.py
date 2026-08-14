"""Speaker Verification endpoints (mounted at /tasks/verification)."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from dataclasses import asdict

from .dataset import (
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .service import (
    SpeakerModelUnavailable,
    UnsupportedSpeakerModel,
    batch_verification_analysis,
    get_model_spec,
    list_models,
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

            return await run_in_threadpool(
                batch_verification_analysis,
                model,
                audio_paths,
                labels,
            )
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
        return await run_in_threadpool(
            batch_verification_analysis,
            payload.model,
            audio_paths,
            recording_ids,
        )
    except (ValueError, RuntimeError) as error:
        status = 503 if isinstance(error, SpeakerModelUnavailable) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error


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