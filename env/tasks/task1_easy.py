"""Task 1 (Easy) – Detect missing values in a synthetic dataset.

The agent must INSPECT the data to find columns with null values, then
VALIDATE to confirm the finding.  FIX is not required for this task.

Correct action sequence (one valid path):
    INSPECT → VALIDATE → done
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from env.data.generator import generate_base_dataset
from env.data.bug_injector import inject_null_values


class Task1Easy:
    """Easy task: find columns that contain null values."""

    TASK_ID = "task1"
    DESCRIPTION = (
        "A dataset has been loaded into the pipeline. "
        "Your goal is to detect all columns that contain missing (null) values. "
        "Use INSPECT to examine the data, then VALIDATE when you have found them."
    )
    # Correct high-level action sequence for full reward
    _OPTIMAL_SEQUENCE = ["INSPECT", "VALIDATE"]

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-initialise the task to its starting state."""
        clean = generate_base_dataset(n_rows=30)
        self._df: pd.DataFrame = inject_null_values(clean, seed=0)
        self._actions_taken: List[str] = []
        self._issues_found: List[str] = []
        self._inspected: bool = False
        self._validated: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _null_columns(self) -> List[str]:
        """Return column names that have at least one null value."""
        return [col for col in self._df.columns if self._df[col].isna().any()]

    def data_sample(self) -> List[Dict[str, Any]]:
        """Return up to 5 rows as a list of dicts."""
        return self._df.head(5).to_dict(orient="records")

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def step(self, action: str) -> Tuple[float, bool, Dict[str, Any]]:
        """Process one agent action.

        Returns:
            (reward, done, info)
        """
        self._actions_taken.append(action)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if action == "INSPECT":
            if not self._inspected:
                # Correct first move
                self._inspected = True
                null_cols = self._null_columns()
                self._issues_found = [
                    f"Column '{c}' has {self._df[c].isna().sum()} null value(s)"
                    for c in null_cols
                ]
                reward = 0.2
                info["hint"] = f"Null columns detected: {null_cols}"
            else:
                # Redundant INSPECT
                reward = -0.1
                info["hint"] = "Data already inspected."

        elif action == "FIX":
            # FIX is not the right approach for detection-only task
            reward = -0.1
            info["hint"] = "Task 1 requires detection, not fixing."

        elif action == "VALIDATE":
            if self._inspected and not self._validated:
                self._validated = True
                reward = 0.2
                done = True
                info["hint"] = "Validation complete. Null columns confirmed."
            elif not self._inspected:
                reward = -0.1
                info["hint"] = "Inspect the data before validating."
            else:
                reward = -0.1
                info["hint"] = "Already validated."

        elif action == "NOOP":
            reward = -0.05
            info["hint"] = "No operation performed."

        return reward, done, info

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def issues_found(self) -> List[str]:
        return list(self._issues_found)

    def is_done(self) -> bool:
        return self._validated
