"""Pydantic models for the OpenEnv interface."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    NOOP = "NOOP"
    FIX_NULL = "FIX_NULL"
    FIX_TYPE = "FIX_TYPE"
    FIX_DUPLICATE = "FIX_DUPLICATE"
    FIX_SCHEMA = "FIX_SCHEMA"
    DEBUG_PIPELINE = "DEBUG_PIPELINE"


class DataAction(BaseModel):
    action_type: ActionType
    target_column: Optional[str] = None
    value: Optional[Any] = None


class Observation(BaseModel):
    step_count: int = Field(ge=0)
    data_snapshot: Dict[str, Any] = Field(default_factory=dict)
    issues_found: List[str] = Field(default_factory=list)
    done: bool = False


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
