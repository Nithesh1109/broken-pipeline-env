"""FastAPI server exposing the OpenEnv interface."""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.core import Environment
from env.models import ActionType, DataAction, GraderResult, Observation, StepResult

app = FastAPI(
    title="broken-pipeline-env",
    description="OpenEnv-compatible environment for LLM agent evaluation.",
    version="1.0.0",
)

# One shared environment instance per server process.
# For production use, replace with a session-keyed store.
_env = Environment()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "task1_easy"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: ActionType
    target_column: Optional[str] = None
    value: Optional[Any] = None


class GraderRequest(BaseModel):
    task_id: Optional[str] = None  # unused — grades the active episode


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/ping")
def ping() -> Dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks."""
    return {"tasks": _env.list_tasks()}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    """Reset the environment to start a new episode."""
    try:
        obs = _env.reset(task_id=request.task_id, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest) -> StepResult:
    """Take one action in the environment."""
    action = DataAction(
        action_type=request.action_type,
        target_column=request.target_column,
        value=request.value,
    )
    try:
        result = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the current environment state."""
    return _env.state()


@app.post("/grader", response_model=GraderResult)
def grader(_request: GraderRequest = GraderRequest()) -> GraderResult:
    """Score the current episode deterministically."""
    return _env.grade()
