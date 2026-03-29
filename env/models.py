from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    INSPECT = "INSPECT"
    RENAME_COLUMN = "RENAME_COLUMN"
    CAST_TYPE = "CAST_TYPE"
    FILL_DEFAULT = "FILL_DEFAULT"
    DROP_COLUMN = "DROP_COLUMN"
    VALIDATE = "VALIDATE"
    MASK_PII = "MASK_PII"
    NOOP = "NOOP"


class DetectedIssue(BaseModel):
    issue_type: str
    column: Optional[str] = None
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class DataAction(BaseModel):
    action_type: ActionType
    target_column: Optional[str] = None
    transformation: Optional[str] = None
    justification: str
    identified_issues: Optional[List[DetectedIssue]] = None


class DataObservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # ONE config block only — never duplicate this

    dataset_preview: List[dict]
    column_schema: dict = Field(alias="schema")
    # alias="schema" preserves wire format without shadowing BaseModel.schema
    pipeline_stage: str
    validation_report: List[DetectedIssue]
    time_remaining: int
    downstream_health: float
    step_count: int
    task_id: int
    pipeline_stage_health: Optional[dict[str, float]] = None


class StepResult(BaseModel):
    observation: DataObservation
    reward: float
    done: bool
    info: dict


class GraderResult(BaseModel):
    score: float
    breakdown: dict[str, float]
    explanation: str