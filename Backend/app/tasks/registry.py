"""
Central backend task registry.

This mirrors the frontend registry (Frontend/src/tasks/registry.tsx) but only
carries what the backend needs: model ids that can be dispatched and dataset
ids that can be resolved. The frontend registry is authoritative for the UI
(names, routes, capabilities); the ONLY contract shared between the two is the
model/dataset id strings.

To add a model or dataset to a task, edit the entry here AND (for models) make
sure a handler exists — either in the legacy shared services
(app/services/model_loader_service.py + MODEL_FUNCTIONS in
app/api/routes/inferences.py) or in your own app/tasks/<task>/service.py.

Task ids are stable and must never change (they appear in route prefixes).
"""

TASKS: dict[str, dict] = {
    "transcription": {
        "name": "Speech Transcription",
        "status": "active",
        # Served by the legacy /inferences/* endpoints (MODEL_FUNCTIONS).
        "models": ["whisper-base", "whisper-large"],
        "datasets": ["common-voice"],
    },
    "emotion": {
        "name": "Emotion Recognition",
        "status": "active",
        "models": ["wav2vec2"],
        "datasets": ["ravdess"],
    },
    "verification": {
        "name": "Speaker Verification",  # rename after the task is finalized
        "status": "active",
        "models": ["ecapa-tdnn", "resnet34-lm"],
        "datasets": ["voxceleb1-indian-demo"],
    },
    "task-b": {
        "name": "Speaker Diarization",
        "status": "active",
        "models": ["pyannote-3.1"],
        "datasets": ["ami-subset"],
    },
    "deepfake": {
        "name": "Audio Deepfake Detection",
        "status": "active",
        "models": ["xlsr-deepfake", "ast-fakeaudio"],
        "datasets": ["asvspoof2019-la"],
    },
}
