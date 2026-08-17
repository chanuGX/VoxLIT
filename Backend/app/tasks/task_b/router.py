"""Speaker Diarization endpoints (mounted at /tasks/task-b)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result

from .dataset import (
    DATASET_ID,
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .service import (
    DiarizationModelUnavailable,
    UnsupportedDiarizationModel,
    file_sha256,
    get_model_spec,
    list_models,
    run_diarization,
)

router = APIRouter()

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # demo files are static; keep for a week


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
        "dataset_id": DATASET_ID,
        "total_recordings": len(recordings),
        "recordings": [asdict(recording) for recording in recordings],
    }


@router.get("/dataset/recordings/{recording_id}")
async def dataset_recording(recording_id: str):
    try:
        recording = get_recording(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return asdict(recording)


@router.get("/dataset/recordings/{recording_id}/audio")
async def dataset_recording_audio(recording_id: str):
    """Serve the audio itself so the frontend player can seek segments."""
    try:
        path = resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="audio/wav", filename=path.name)


class RunRequest(BaseModel):
    model: str
    recording_id: str


async def _get_or_compute(model_key: str, recording_id: str) -> dict:
    """Cache-through diarization; key = model + content hash of the file."""
    try:
        get_model_spec(model_key)
    except UnsupportedDiarizationModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        path = resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"diar:{model_key}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_diarization, model_key, path)
    except DiarizationModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": recording_id, "cached": False}


@router.post("/run")
async def run(request: RunRequest):
    return await _get_or_compute(request.model, request.recording_id)


@router.get("/projection")
async def projection(model: str, recording_id: str):
    """2D PCA of segment embeddings → [{id, x, y, speaker, confidence}]."""
    result = await _get_or_compute(model, recording_id)

    embeddings = result.get("embeddings", {})
    if len(embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail="Not enough embeddable segments for a 2D projection.",
        )

    def _pca() -> list[dict]:
        import numpy as np
        from sklearn.decomposition import PCA

        segment_ids = list(embeddings.keys())
        matrix = np.asarray([embeddings[i] for i in segment_ids], dtype=np.float32)
        coords = PCA(n_components=2).fit_transform(matrix)
        by_id = {s["id"]: s for s in result["segments"]}
        return [
            {
                "id": segment_id,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "speaker": by_id[segment_id]["speaker"],
                "confidence": by_id[segment_id]["confidence"],
            }
            for segment_id, (x, y) in zip(segment_ids, coords)
        ]

    return {
        "model": model,
        "recording_id": recording_id,
        "points": await run_in_threadpool(_pca),
    }