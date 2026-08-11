"""
VoxLIT task packages.

Each task lives in its own subpackage (owned by one team member) and exposes:
  * TASK_INFO: {"id": ..., "name": ...} — id must match registry.py
  * router (optional): a FastAPI APIRouter mounted at /tasks/<id> by main.py

To register a new task: create app/tasks/<your_task>/ with an __init__.py
exporting TASK_INFO (and a router if you have endpoints), then add the module
to ACTIVE_TASK_MODULES below and an entry to registry.py. Those two one-line
edits are the only shared-file changes a new task needs.
"""
from .router import router  # GET /tasks
from . import transcription, emotion, verification, task_b, task_c

ACTIVE_TASK_MODULES = [transcription, emotion, verification, task_b, task_c]

__all__ = ["router", "ACTIVE_TASK_MODULES"]
