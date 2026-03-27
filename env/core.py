"""Core Environment class implementing the OpenEnv interface."""

from typing import Any, Dict, List, Optional

from env.models import ActionType, DataAction, GraderResult, Observation, StepResult

MAX_STEPS = 8

_TASK_REGISTRY = {
    "task1_easy": ("env.tasks.task1_easy", "env.graders.grader1"),
    "task2_medium": ("env.tasks.task2_medium", "env.graders.grader2"),
    "task3_hard": ("env.tasks.task3_hard", "env.graders.grader3"),
}


def _import(module_path: str):
    import importlib
    return importlib.import_module(module_path)


class Environment:
    """OpenEnv-compatible environment for data pipeline debugging tasks.

    Usage
    -----
    env = Environment()
    obs = env.reset(task_id="task1_easy")
    result = env.step(DataAction(action_type=ActionType.FIX_NULL, target_column="age"))
    score = env.grade()
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._state: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []
        self._task_module = None
        self._grader_module = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1_easy", seed: int = 42) -> Observation:
        """Initialise a new episode for *task_id*."""
        if task_id not in _TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(_TASK_REGISTRY)}")
        task_mod_path, grader_mod_path = _TASK_REGISTRY[task_id]
        self._task_module = _import(task_mod_path)
        self._grader_module = _import(grader_mod_path)
        self._task_id = task_id
        self._state = self._task_module.build_initial_state(seed=seed)
        self._step_count = 0
        self._done = False
        self._history = []
        return self._build_observation()

    def step(self, action: DataAction) -> StepResult:
        """Apply *action* and return the resulting :class:`StepResult`."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"msg": "Episode already done"})

        reward, info, task_done = self._task_module.evaluate_action(
            self._state,
            action.action_type,
            action.target_column,
            action.value,
        )

        self._step_count += 1
        self._history.append(
            {
                "step": self._step_count,
                "action": action.model_dump(),
                "reward": reward,
                "info": info,
            }
        )

        self._done = task_done or self._step_count >= MAX_STEPS
        obs = self._build_observation()
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current episode state."""
        if self._state is None:
            return {"status": "not_started"}
        from env.data.generator import dataset_to_dict
        snap = dataset_to_dict(self._state["df"]) if "df" in self._state else {}
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "done": self._done,
            "data_snapshot": snap,
            "issues": self._collect_issues(),
        }

    def grade(self) -> GraderResult:
        """Score the current episode deterministically."""
        if self._state is None or self._grader_module is None:
            return GraderResult(score=0.0, details={"msg": "No active episode"})
        return self._grader_module.grade(self._state, self._history)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered tasks."""
        import importlib
        tasks = []
        for tid, (task_mod_path, _) in _TASK_REGISTRY.items():
            mod = importlib.import_module(task_mod_path)
            tasks.append(
                {
                    "task_id": tid,
                    "name": mod.TASK_NAME,
                    "difficulty": mod.DIFFICULTY,
                    "max_steps": MAX_STEPS,
                }
            )
        return tasks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        from env.data.generator import dataset_to_dict
        snap = dataset_to_dict(self._state["df"]) if self._state and "df" in self._state else {}
        return Observation(
            step_count=self._step_count,
            data_snapshot=snap,
            issues_found=self._collect_issues(),
            done=self._done,
        )

    def _collect_issues(self) -> List[str]:
        if self._state is None:
            return []
        issues = []
        null_cols = self._state.get("null_columns") or [
            col for col, cnt in self._state.get("null_counts", {}).items() if cnt > 0
        ]
        issues.extend(f"null:{c}" for c in null_cols)
        type_cols = self._state.get("type_error_columns") or self._state.get("type_errors", {})
        issues.extend(f"type_error:{c}" for c in type_cols)
        if self._state.get("has_duplicates"):
            issues.append("duplicates")
        return issues
