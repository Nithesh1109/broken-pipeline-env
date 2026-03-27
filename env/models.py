"""Pydantic data models for the OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All actions an agent can take in one step."""
    INSPECT = "INSPECT"
    FIX = "FIX"
    VALIDATE = "VALIDATE"
    NOOP = "NOOP"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DataAction(BaseModel):
    """Wrapper sent to POST /step."""
    action: ActionType = Field(..., description="Action chosen by the agent.")


class Observation(BaseModel):
    """What the environment returns after reset or step."""
    task_id: str = Field(..., description="Identifier of the active task.")
    step: int = Field(..., description="Current step number (0-indexed).")
    max_steps: int = Field(default=8, description="Maximum steps per episode.")
    description: str = Field(..., description="Human-readable task description.")
    data_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample rows from the dataset (up to 5 rows).",
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="Issues discovered so far in this episode.",
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint surfaced after a correct INSPECT.",
    )


class StepResult(BaseModel):
    """Full result returned by POST /step."""
    observation: Observation
    reward: float = Field(..., description="Reward for this step.")
    done: bool = Field(..., description="Whether the episode has ended.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information.",
    )


class GraderResult(BaseModel):
    """Output from GET /grader."""
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Final score [0, 1].")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Score components keyed by criterion.",
    )
    explanation: str = Field(..., description="Human-readable grading rationale.")
