"""
Data models for the broken-pipeline environment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """An observation returned by the environment at each step."""

    task_id: str
    step: int
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self.step,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """An action submitted by the agent."""

    action_type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "payload": self.payload,
        }


@dataclass
class StepResult:
    """The result of executing one environment step."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    task_id: str
    name: str
    description: str
    max_steps: int = 10
    scenario_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Summary of a completed episode."""

    task_id: str
    total_steps: int
    total_reward: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "success": self.success,
            "details": self.details,
        }
