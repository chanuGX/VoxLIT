"""Voice Task B — OWNER: member 2.

Implement your endpoints in router.py and your model loading/inference in
service.py. The router is mounted at /tasks/task-b by main.py.
"""
from .router import router

TASK_INFO = {"id": "task-b", "name": "Voice Task B"}

__all__ = ["TASK_INFO", "router"]
