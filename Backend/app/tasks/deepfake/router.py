"""Audio Deepfake Detection endpoints (mounted at /tasks/deepfake)."""

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
from .evaluation import evaluate_dataset
from .saliency import METHOD as SALIENCY_METHOD, SaliencyUnavailable, generate_saliency
from .silence_probe import SILENCE_TOP_DB, run_silence_probe
from .metrics import NotEnoughLabelledData
from .service import (
    THRESHOLD_VERSION,
    DeepfakeModelUnavailable,
    UnsupportedDeepfakeModel,
    file_sha256,
    get_model_spec,
    list_models,
    run_detection,
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
    """Opaque recording ids only -- never the bona fide/spoof key. See
    dataset.py's module docstring."""
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
    """Serve the audio itself so the frontend player can listen to the clip."""
    try:
        path = resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="audio/flac", filename=path.name)


class EvaluationRequest(BaseModel):
    model: str


@router.post("/scores")
async def evaluation(request: EvaluationRequest):
    """Feature 1 — score distributions, DET curve and EER (SRS DF-6..DF-9).

    Aggregates only: no per-recording label is ever returned, so the
    workbench's read-the-score-then-check-the-protocol exercise survives.

    The first call scores the whole dataset and can take minutes; per-clip
    scores are cached afterwards, shared with /run.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        return await evaluate_dataset(request.model)
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except NotEnoughLabelledData as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


class SilenceProbeRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/silence-probe")
async def silence_probe(request: SilenceProbeRequest):
    """Feature 2 — score the clip as submitted, trimmed, and silence-only.

    SRS DF-10..DF-12. Three forward passes on one clip: no dataset, no
    labels, no gradients. Cached on the model, the silence threshold and the
    file's content hash, so re-opening the card is free.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        path = resolve_recording_path(request.recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df-silence:{request.model}:{THRESHOLD_VERSION}:{SILENCE_TOP_DB}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_silence_probe, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}


class SaliencyRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/saliency")
async def saliency(request: SaliencyRequest):
    """Feature 3 — waveform-aligned temporal attribution (SRS DF-14, DF-15).

    One forward and one backward pass over a single clip. Emits the shared
    saliency service's response contract so the payload is interchangeable
    with its visualisation.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        path = resolve_recording_path(request.recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df-saliency:{request.model}:{SALIENCY_METHOD}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(generate_saliency, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SaliencyUnavailable as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}


class RunRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/run")
async def run(request: RunRequest):
    """Cache-through detection; key = model + threshold version + file hash."""
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        path = resolve_recording_path(request.recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df:{request.model}:{THRESHOLD_VERSION}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_detection, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}
