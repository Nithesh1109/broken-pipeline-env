from __future__ import annotations

from pathlib import Path

import pandas as pd

from env.data.bug_injector import inject_bugs, load_scenario
from env.data.generator import generate_employee_dataset
from env.models import ActionType, DataAction, DataObservation, DetectedIssue, StepResult


class Task3IncidentEnv:
    """Task 3 environment for full data incident response handling."""

    MAX_STEPS = 8
    SCENARIO_PATH = Path(__file__).parent.parent / "data" / "scenarios" / "task3_scenario.json"
    CORRECT_DIAGNOSIS_KEYWORDS = [
        "stage 3",
        "join stage",
        "schema drift",
        "ssn",
        "pii",
        "type mismatch",
        "revenue",
        "aggregation",
    ]

    def __init__(self) -> None:
        """Initialize mutable state containers for incident simulation."""
        self.df: pd.DataFrame = pd.DataFrame()
        self.ground_truth: list[dict] = []
        self.step_count: int = 0

        self.diagnosis_correct: bool = False
        self.fix_applied: bool = False
        self.pii_masked: bool = False
        self.validation_passed: bool = False

        self.pipeline_stage_health: dict[str, float] = {}
        self.downstream_health: float = 0.0

    def reset(self) -> DataObservation:
        """Reset state, build deterministic dataset, inject bugs, and return observation."""
        clean_df = generate_employee_dataset(seed=42)
        clean_df["revenue_amount"] = (clean_df["salary"].astype(float) * 1.35).round(2)
        scenario_bugs = load_scenario(str(self.SCENARIO_PATH))
        self.df, self.ground_truth = inject_bugs(clean_df, scenario_bugs)

        self.step_count = 0
        self.diagnosis_correct = False
        self.fix_applied = False
        self.pii_masked = False
        self.validation_passed = False

        self.pipeline_stage_health = {
            "stage_1_ingest": 1.0,
            "stage_2_clean": 1.0,
            "stage_3_join": 0.0,
            "stage_4_aggregate": 0.3,
            "stage_5_output": 0.0,
        }
        self.downstream_health = sum(self.pipeline_stage_health.values()) / 5
        return self._build_observation()

    def step(self, action: DataAction) -> StepResult:
        """Apply one incident response action and return the resulting transition."""
        reward = 0.0
        done = False

        justification_lower = action.justification.lower()
        keyword_hits = sum(1 for kw in self.CORRECT_DIAGNOSIS_KEYWORDS if kw in justification_lower)
        target_relevant = action.target_column in ["rev_amt", "revenue_amount", "ssn", None]

        if action.action_type == ActionType.INSPECT:
            if keyword_hits >= 2 and target_relevant:
                self.diagnosis_correct = True
                self.pipeline_stage_health["stage_3_join"] = 0.5
                reward += 0.25
            elif keyword_hits >= 1:
                reward += min(0.05 * keyword_hits, 0.15)

        elif action.action_type == ActionType.RENAME_COLUMN:
            if action.target_column == "rev_amt":
                if "rev_amt" in self.df.columns and "revenue_amount" not in self.df.columns:
                    self.df.rename(columns={"rev_amt": "revenue_amount"}, inplace=True)
                reward += 0.15
                if self.diagnosis_correct:
                    reward += 0.05
            else:
                reward -= 0.10

        elif action.action_type == ActionType.CAST_TYPE:
            if action.target_column in ["rev_amt", "revenue_amount"] and action.transformation == "cast_to_float":
                col = "revenue_amount" if "revenue_amount" in self.df.columns else "rev_amt"
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(float)
                    self.fix_applied = True
                    reward += 0.20
                    self.pipeline_stage_health["stage_3_join"] = 1.0
                    self.pipeline_stage_health["stage_4_aggregate"] = 0.8
                else:
                    reward -= 0.10
            else:
                reward -= 0.10

        elif action.action_type == ActionType.MASK_PII:
            if action.target_column == "ssn" and "ssn" in self.df.columns:
                self.df["ssn"] = self.df["ssn"].astype(str).str.replace(r"\d", "X", regex=True)
                self.pii_masked = True
                reward += 0.20
            else:
                reward -= 0.10

        elif action.action_type == ActionType.VALIDATE:
            if self.fix_applied and self.pii_masked:
                self.validation_passed = True
                self.pipeline_stage_health["stage_4_aggregate"] = 1.0
                self.pipeline_stage_health["stage_5_output"] = 1.0
                reward += 0.30
                done = True
            else:
                stages_above_half = sum(1 for v in self.pipeline_stage_health.values() if v >= 0.5)
                reward += 0.05 * (stages_above_half / 5)

        elif action.action_type == ActionType.NOOP:
            reward = 0.0

        else:
            reward -= 0.10

        self.downstream_health = sum(self.pipeline_stage_health.values()) / 5
        reward = max(-0.5, min(1.0, reward))
        self.step_count += 1
        done = done or (self.step_count >= self.MAX_STEPS)

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "diagnosis_correct": self.diagnosis_correct,
                "fix_applied": self.fix_applied,
                "pii_masked": self.pii_masked,
                "validation_passed": self.validation_passed,
            },
        )

    def _build_observation(self) -> DataObservation:
        """Build observation for current task state and unresolved issue list."""
        schema_dict = {
            col: {"type": str(dtype), "nullable": bool(self.df[col].isna().any())}
            for col, dtype in self.df.dtypes.items()
        }

        validation_report = [
            DetectedIssue(
                issue_type=truth["type"],
                column=truth.get("column"),
                description=truth["description"],
                severity=truth["severity"],
            )
            for truth in self.ground_truth
        ]

        return DataObservation(
            dataset_preview=self.df.head(10).to_dict(orient="records"),
            column_schema=schema_dict,
            pipeline_stage="INCIDENT_RESPONSE",
            validation_report=validation_report,
            time_remaining=self.MAX_STEPS - self.step_count,
            downstream_health=self.downstream_health,
            step_count=self.step_count,
            task_id=3,
            pipeline_stage_health=dict(self.pipeline_stage_health),
        )

    def state(self) -> DataObservation:
        """Return current observation snapshot without side effects."""
        return self._build_observation()
