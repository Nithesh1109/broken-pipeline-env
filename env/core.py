"""env/core.py – OpenEnvEnvironment: the central environment controller.

Implements the OpenEnv-style interface:
    reset(task_id)  → Observation
    step(action)    → StepResult
    state()         → dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env.models import ActionType, Observation, StepResult
from env.tasks.task1_easy import Task1Easy
from env.tasks.task2_medium import Task2Medium
from env.tasks.task3_hard import Task3Hard
from env.graders import grader1, grader2, grader3


# Registry: task_id → (task class, grader module)
_TASK_REGISTRY: Dict[str, Any] = {
    "task1": (Task1Easy, grader1),
    "task2": (Task2Medium, grader2),
    "task3": (Task3Hard, grader3),
}

MAX_STEPS = 8

# Reward constants (also defined in openenv.yaml for documentation)
REWARD_CORRECT = 0.2
REWARD_WRONG = -0.1
REWARD_NOOP = -0.05


class OpenEnvEnvironment:
    """Central environment controller.

    Usage
    -----
    env = OpenEnvEnvironment()
    obs = env.reset("task1")
    result = env.step("INSPECT")
    grader_output = env.grade()
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._task = None
        self._grader_module = None
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._actions_taken: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        task_id:
            One of ``"task1"``, ``"task2"``, ``"task3"``.
        """
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid options: {list(_TASK_REGISTRY)}"
            )

        task_cls, grader_mod = _TASK_REGISTRY[task_id]
        self._task_id = task_id
        self._task = task_cls()
        self._grader_module = grader_mod
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._actions_taken = []

        return self._build_observation()

    def step(self, action: str) -> StepResult:
        """Apply an action and return the step result.

        Parameters
        ----------
        action:
            One of the ``ActionType`` enum values as a string.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is already finished. Call reset() to start a new one.")

        # Normalise and validate action string
        try:
            action_enum = ActionType(action.upper())
        except ValueError:
            action_enum = ActionType.NOOP

        action_str = action_enum.value
        self._actions_taken.append(action_str)
        self._step_count += 1

        # Delegate to task
        reward, task_done, info = self._task.step(action_str)
        self._cumulative_reward += reward

        # Episode ends on task completion or max steps reached
        if task_done or self._step_count >= MAX_STEPS:
            self._done = True

        obs = self._build_observation(hint=info.get("hint"))

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "step": self._step_count,
                **info,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return the full current state as a plain dict."""
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "max_steps": MAX_STEPS,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "actions_taken": list(self._actions_taken),
            "issues_found": self._task.issues_found() if self._task else [],
            "data_sample": self._task.data_sample() if self._task else [],
        }

    def grade(self) -> Dict[str, Any]:
        """Run the grader for the current task and return the result dict."""
        if self._grader_module is None:
            raise RuntimeError("Call reset() before grading.")
        result = self._grader_module.grade(self.state())
        return result.model_dump()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, hint: Optional[str] = None) -> Observation:
        """Construct an Observation from the current task state."""
        return Observation(
            task_id=self._task_id or "",
            step=self._step_count,
            max_steps=MAX_STEPS,
            description=self._task.DESCRIPTION,
            data_sample=self._task.data_sample(),
            issues_found=self._task.issues_found(),
            hint=hint,
        )
