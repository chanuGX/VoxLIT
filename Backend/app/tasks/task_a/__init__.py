"""Voice Task A — OWNER: member 1.

Implement your endpoints in router.py and your model loading/inference in
service.py. The router is mounted at /tasks/task-a by main.py.
"""
from .router import router

TASK_INFO = {"id": "task-a", "name": "Voice Task A"}

__all__ = ["TASK_INFO", "router"]
