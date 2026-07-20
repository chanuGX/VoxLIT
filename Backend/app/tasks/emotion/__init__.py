"""Emotion Recognition task.

Uses the legacy shared endpoints (/inferences/*, /saliency/generate, /perturb)
— no task-scoped router needed. Model handlers live in
app/services/model_loader_service.py via MODEL_FUNCTIONS in
app/api/routes/inferences.py.
"""

TASK_INFO = {"id": "emotion", "name": "Emotion Recognition"}
