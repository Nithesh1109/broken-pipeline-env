"""
Core environment logic for broken-pipeline-env.
"""

import json
import os
from typing import Any, Dict, Optional

from env.models import Action, EpisodeResult, Observation, StepResult, TaskConfig
from env.tasks.task1_audit import AuditTask
from env.tasks.task2_schema import SchemaTask
from env.tasks.task3_incident import IncidentTask

_TASK_REGISTRY = {
    "task1_audit": AuditTask,
    "task2_schema": SchemaTask,
    "task3_incident": IncidentTask,
}


class BrokenPipelineEnv:
    """
    Main environment class. Supports three tasks:
      - task1_audit   : Identify anomalies in pipeline audit logs.
      - task2_schema  : Detect and report schema violations in records.
      - task3_incident: Diagnose the root cause of a pipeline incident.
    """

    def __init__(self, task_id: str, scenario_path: Optional[str] = None):
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(_TASK_REGISTRY.keys())}"
            )
        self.task_id = task_id
        self.scenario_path = scenario_path or self._default_scenario_path(task_id)
        self._task = None
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        scenario = self._load_scenario(self.scenario_path)
        task_cls = _TASK_REGISTRY[self.task_id]
        self._task = task_cls(scenario)
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        return self._task.get_observation(step=0)

    def step(self, action: Action) -> StepResult:
        """Execute one step in the environment."""
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new one.")

        self._step_count += 1
        reward, done, info = self._task.evaluate_action(action, self._step_count)
        self._total_reward += reward
        self._done = done or (self._step_count >= self._task.max_steps)
        obs = self._task.get_observation(step=self._step_count)
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def get_episode_result(self) -> EpisodeResult:
        """Return summary statistics for the current episode."""
        if self._task is None:
            raise RuntimeError("Call reset() before get_episode_result().")
        return EpisodeResult(
            task_id=self.task_id,
            total_steps=self._step_count,
            total_reward=self._total_reward,
            success=self._task.is_solved(),
            details=self._task.get_details(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_scenario_path(task_id: str) -> str:
        here = os.path.dirname(__file__)
        # Scenario files are named task1_scenario.json, task2_scenario.json, etc.
        # Strip the descriptive suffix (e.g. "task1_audit" -> "task1").
        short_id = task_id.split("_")[0]
        filename = f"{short_id}_scenario.json"
        return os.path.join(here, "data", "scenarios", filename)

    @staticmethod
    def _load_scenario(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
