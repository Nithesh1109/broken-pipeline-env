from __future__ import annotations

from pathlib import Path

import pandas as pd

from env.data.bug_injector import inject_bugs, load_scenario, matches_ground_truth
from env.data.generator import generate_employee_dataset
from env.models import ActionType, DataAction, DataObservation, DetectedIssue, StepResult


class Task1AuditEnv:
    """Task 1 environment for data quality audit and direct remediation."""

    MAX_STEPS = 8
    TOTAL_BUGS = 5
    SCENARIO_PATH = Path(__file__).parent.parent / "data" / "scenarios" / "task1_scenario.json"

    def __init__(self) -> None:
        """Initialize Task1 mutable state containers."""
        self.df: pd.DataFrame = pd.DataFrame()
        self.ground_truth: list[dict] = []
        self.step_count: int = 0
        self.identified_bug_ids: set[str] = set()
        self.fixed_bug_ids: set[str] = set()
        self.downstream_health: float = 0.0

    def reset(self) -> DataObservation:
        """Reset state, generate deterministic data, inject bugs, and return observation."""
        scenario = load_scenario(str(self.SCENARIO_PATH))
        clean_df = generate_employee_dataset(seed=42)
        self.df, self.ground_truth = inject_bugs(clean_df, scenario)
        self.step_count = 0
        self.identified_bug_ids = set()
        self.fixed_bug_ids = set()
        self.downstream_health = 0.0
        return self._build_observation()

    def step(self, action: DataAction) -> StepResult:
        """Apply an agent action and return the resulting transition tuple."""
        reward = 0.0
        done = False

        if action.action_type == ActionType.INSPECT:
            for issue in (action.identified_issues or []):
                for truth in self.ground_truth:
                    if matches_ground_truth(issue, truth) and truth["bug_id"] not in self.identified_bug_ids:
                        reward += 0.15
                        self.identified_bug_ids.add(truth["bug_id"])
                        break
                else:
                    reward -= 0.05

        elif (
            action.action_type == ActionType.FILL_DEFAULT
            and action.target_column == "salary"
            and action.transformation == "fill_median"
        ):
            if "B001" not in self.fixed_bug_ids:
                median_val = self.df["salary"].median()
                self.df["salary"] = self.df["salary"].fillna(median_val)
                reward += 0.20
                self.fixed_bug_ids.add("B001")
            else:
                reward -= 0.05

        elif (
            action.action_type == ActionType.CAST_TYPE
            and action.target_column == "age"
            and action.transformation == "cast_to_int"
        ):
            action_fixed_any = False
            if "B002" not in self.fixed_bug_ids:
                self.df.loc[5, "age"] = 23
                self.fixed_bug_ids.add("B002")
                reward += 0.20
                action_fixed_any = True

            numeric_age = pd.to_numeric(self.df["age"], errors="coerce")
            invalid_mask = (numeric_age > 150) | (numeric_age < 0)
            if invalid_mask.any() and "B003" not in self.fixed_bug_ids:
                median_age = int(numeric_age[(numeric_age >= 0) & (numeric_age <= 150)].median())
                self.df.loc[invalid_mask, "age"] = median_age
                self.fixed_bug_ids.add("B003")
                reward += 0.20
                action_fixed_any = True

            if not action_fixed_any:
                reward -= 0.05

        elif action.action_type == ActionType.VALIDATE:
            action_fixed_any = False
            if "B004" not in self.fixed_bug_ids:
                current = str(self.df.loc[10, "phone"])
                digits = "".join(ch for ch in current if ch.isdigit())
                if len(digits) >= 10:
                    self.df.loc[10, "phone"] = digits[-10:]
                self.fixed_bug_ids.add("B004")
                action_fixed_any = True

            if "B005" not in self.fixed_bug_ids:
                self.df = self.df.drop_duplicates().reset_index(drop=True)
                self.fixed_bug_ids.add("B005")
                action_fixed_any = True

            if not action_fixed_any:
                reward -= 0.05

            fixed_ratio = len(self.fixed_bug_ids) / self.TOTAL_BUGS
            if len(self.fixed_bug_ids) == self.TOTAL_BUGS:
                reward += 0.30
                done = True
            else:
                reward += 0.10 * fixed_ratio

        elif action.action_type == ActionType.DROP_COLUMN:
            reward -= 0.10

        elif action.action_type == ActionType.NOOP:
            reward = 0.0

        reward = max(-0.5, min(1.0, reward))
        self.step_count += 1
        self.downstream_health = len(self.fixed_bug_ids) / self.TOTAL_BUGS
        done = done or (self.step_count >= self.MAX_STEPS)

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "fixed": list(self.fixed_bug_ids),
                "identified": list(self.identified_bug_ids),
            },
        )

    def _build_observation(self) -> DataObservation:
        """Construct a DataObservation from current in-memory state."""
        unfixed_bugs = [t for t in self.ground_truth if t["bug_id"] not in self.fixed_bug_ids]
        validation_report = [
            DetectedIssue(
                issue_type=b["type"],
                column=b.get("column"),
                description=b["description"],
                severity=b["severity"],
            )
            for b in unfixed_bugs
        ]

        schema_dict = {
            col: {
                "type": str(dtype),
                "nullable": bool(self.df[col].isna().any()),
            }
            for col, dtype in self.df.dtypes.items()
        }

        return DataObservation(
            dataset_preview=self.df.head(10).to_dict(orient="records"),
            column_schema=schema_dict,
            pipeline_stage="AUDIT",
            validation_report=validation_report,
            time_remaining=self.MAX_STEPS - self.step_count,
            downstream_health=self.downstream_health,
            step_count=self.step_count,
            task_id=1,
            pipeline_stage_health=None,
        )

    def state(self) -> DataObservation:
        """Return current observation snapshot without side effects."""
        return self._build_observation()
