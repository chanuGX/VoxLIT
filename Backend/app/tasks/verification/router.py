"""Speaker Verification endpoints (mounted at /tasks/verification)."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .service import (
    SpeakerModelUnavailable,
    UnsupportedSpeakerModel,
    get_model_spec,
    list_models,
    temporal_occlusion_saliency,
    verify_speaker,
)

router = APIRouter()

@router.get("/models")
async def available_models():
    return {"models": list_models()}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_BYTES = 50 * 1024 * 1024


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