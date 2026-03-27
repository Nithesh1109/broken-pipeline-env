"""env/server.py – FastAPI application exposing the OpenEnv HTTP interface.

Endpoints
---------
GET  /ping      – Health check
GET  /tasks     – List available tasks
POST /reset     – Reset environment with a chosen task
POST /step      – Take one action in the environment
GET  /state     – Retrieve current environment state
GET  /grader    – Score the current episode
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.core import OpenEnvEnvironment
from env.models import ActionType, DataAction, GraderResult, Observation, StepResult

app = FastAPI(
    title="broken-pipeline-env",
    description="OpenEnv-style environment for LLM agent training.",
    version="1.0.0",
)

# Single shared environment instance (stateful per process)
_env = OpenEnvEnvironment()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping() -> Dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """Return metadata for all available tasks."""
    return {
        "tasks": [
            {
                "task_id": "task1",
                "name": "Detect Missing Values",
                "difficulty": "easy",
            },
            {
                "task_id": "task2",
                "name": "Fix Schema / Format Problems",
                "difficulty": "medium",
            },
            {
                "task_id": "task3",
                "name": "Full Pipeline Debug",
                "difficulty": "hard",
            },
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest) -> Observation:
    """Reset the environment for the specified task."""
    try:
        obs = _env.reset(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return obs


@app.post("/step", response_model=StepResult)
def step(body: DataAction) -> StepResult:
    """Take one action in the current episode."""
    try:
        result = _env.step(body.action.value)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the full current environment state."""
    return _env.state()


@app.get("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    """Grade the current episode and return a structured score."""
    try:
        result = _env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GraderResult(**result)
