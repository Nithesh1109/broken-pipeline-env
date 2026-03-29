from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from env.data.bug_injector import inject_task1_bugs, inject_task2_bugs, inject_task3_bugs
from env.data.generator import generate_employee_dataset
from env.graders.grader1 import grade as grade_task1
from env.graders.grader2 import grade as grade_task2
from env.graders.grader3 import grade as grade_task3
from env.models import DataAction, DataObservation, GraderResult
from env.tasks.task1_audit import run_task as run_task1
from env.tasks.task2_schema import run_task as run_task2
from env.tasks.task3_incident import run_task as run_task3


app = FastAPI(title="broken-pipeline-env", version="1.0.0")

SCENARIO_DIR = Path(__file__).parent / "data" / "scenarios"
TASK_IDS = ["task1", "task2", "task3"]

STATE: dict[str, dict[str, Any]] = {}


def _load_scenario(task_id: str) -> dict[str, Any]:
    file_name = f"{task_id}_scenario.json"
    scenario_file = SCENARIO_DIR / file_name
    if not scenario_file.exists():
        return {"task_id": task_id, "description": "No scenario file found"}
    with scenario_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_task(task_id: str) -> None:
    if task_id not in TASK_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown task id: {task_id}")


def _build_state(task_id: str) -> dict[str, Any]:
    seed = {"task1": 42, "task2": 84, "task3": 126}[task_id]
    clean = generate_employee_dataset(seed=seed)

    if task_id == "task1":
        broken, truth = inject_task1_bugs(clean)
    elif task_id == "task2":
        broken, truth = inject_task2_bugs(clean)
    else:
        broken, truth = inject_task3_bugs(clean)

    return {
        "dataset": broken,
        "ground_truth": truth,
        "scenario": _load_scenario(task_id),
        "observation": None,
    }


def _run_observation(task_id: str, dataset: list[dict]) -> DataObservation:
    if task_id == "task1":
        return run_task1(dataset)
    if task_id == "task2":
        return run_task2(dataset)
    return run_task3(dataset)


def _grade(task_id: str, action: DataAction, obs: DataObservation, truth: dict) -> GraderResult:
    if task_id == "task1":
        return grade_task1(action, obs, truth)
    if task_id == "task2":
        return grade_task2(action, obs, truth)
    return grade_task3(action, obs, truth)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> dict[str, list[str]]:
    return {"tasks": TASK_IDS}


@app.post("/reset/{task_id}")
def reset_task(task_id: str) -> dict[str, Any]:
    _ensure_task(task_id)
    STATE[task_id] = _build_state(task_id)
    return {
        "task_id": task_id,
        "message": "task state reset",
        "scenario": STATE[task_id]["scenario"],
    }


@app.get("/scenario/{task_id}")
def get_scenario(task_id: str) -> dict[str, Any]:
    _ensure_task(task_id)
    if task_id not in STATE:
        STATE[task_id] = _build_state(task_id)
    return {"task_id": task_id, "scenario": STATE[task_id]["scenario"]}


@app.get("/observe/{task_id}", response_model=DataObservation)
def observe(task_id: str) -> DataObservation:
    _ensure_task(task_id)
    if task_id not in STATE:
        STATE[task_id] = _build_state(task_id)
    obs = _run_observation(task_id, STATE[task_id]["dataset"])
    STATE[task_id]["observation"] = obs
    return obs


@app.post("/act/{task_id}", response_model=GraderResult)
def act(task_id: str, action: DataAction) -> GraderResult:
    _ensure_task(task_id)
    if action.task_id != task_id:
        raise HTTPException(status_code=400, detail="task_id in payload must match path")

    if task_id not in STATE:
        STATE[task_id] = _build_state(task_id)

    obs = STATE[task_id]["observation"]
    if obs is None:
        obs = _run_observation(task_id, STATE[task_id]["dataset"])
        STATE[task_id]["observation"] = obs

    return _grade(task_id, action, obs, STATE[task_id]["ground_truth"])