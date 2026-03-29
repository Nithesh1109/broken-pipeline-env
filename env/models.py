from typing import Any

from pydantic import BaseModel, Field


class DataAction(BaseModel):
    task_id: str
    findings: list[str] = Field(default_factory=list)
    remediations: list[str] = Field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataObservation(BaseModel):
    task_id: str
    status: str
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: list[str] = Field(default_factory=list)