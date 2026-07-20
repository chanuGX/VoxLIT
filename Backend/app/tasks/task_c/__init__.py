"""Voice Task C — OWNER: member 3.

Implement your endpoints in router.py and your model loading/inference in
service.py. The router is mounted at /tasks/task-c by main.py.
"""
from .router import router

TASK_INFO = {"id": "task-c", "name": "Voice Task C"}

__all__ = ["TASK_INFO", "router"]
