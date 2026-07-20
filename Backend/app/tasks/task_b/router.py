"""Voice Task B endpoints (mounted at /tasks/task-b)."""
from fastapi import APIRouter

router = APIRouter()


@router.post("/run")
async def run_inference():
    """Replace with real inference once the task's model is added (service.py)."""
    return {"status": "not_implemented", "task": "task-b"}
