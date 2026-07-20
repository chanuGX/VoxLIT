"""Informational task-registry endpoint.

GET /tasks returns the backend's view of the task registry. The frontend does
NOT depend on this (its own registry in Frontend/src/tasks/registry.tsx is
authoritative for the UI) — this endpoint exists for debugging and for keeping
the two registries easy to diff.
"""
from fastapi import APIRouter

from .registry import TASKS

router = APIRouter()


@router.get("/tasks")
async def list_tasks():
    return {"tasks": [{"id": task_id, **info} for task_id, info in TASKS.items()]}
