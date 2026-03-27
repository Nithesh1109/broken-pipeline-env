"""Task 2 (Medium) – Fix schema / format problems in a synthetic dataset.

The dataset has type errors: numeric columns were corrupted with string
suffixes.  The agent must:
    1. INSPECT  – discover the type violations
    2. FIX      – correct the corrupt values
    3. VALIDATE – confirm the schema is clean

Correct action sequence (one valid path):
    INSPECT → FIX → VALIDATE → done
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from env.data.generator import generate_base_dataset
from env.data.bug_injector import inject_type_errors


class Task2Medium:
    """Medium task: detect and fix schema/format problems."""

    TASK_ID = "task2"
    DESCRIPTION = (
        "Some numeric columns in the dataset contain string values due to a "
        "schema drift during ingestion. "
        "Use INSPECT to find them, FIX to correct them, and VALIDATE to confirm."
    )

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        clean = generate_base_dataset(n_rows=30)
        self._df: pd.DataFrame = inject_type_errors(clean, seed=1)
        self._actions_taken: List[str] = []
        self._issues_found: List[str] = []
        self._inspected: bool = False
        self._fixed: bool = False
        self._validated: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bad_rows(self) -> int:
        """Count rows where 'salary' cannot be parsed as a number."""
        return pd.to_numeric(self._df["salary"], errors="coerce").isna().sum()

    def data_sample(self) -> List[Dict[str, Any]]:
        return self._df.head(5).to_dict(orient="records")

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def step(self, action: str) -> Tuple[float, bool, Dict[str, Any]]:
        self._actions_taken.append(action)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if action == "INSPECT":
            if not self._inspected:
                self._inspected = True
                bad = self._bad_rows()
                self._issues_found = [
                    f"'salary' column has {bad} non-numeric value(s)"
                ]
                reward = 0.2
                info["hint"] = f"Found {bad} row(s) with invalid salary format."
            else:
                reward = -0.1
                info["hint"] = "Already inspected."

        elif action == "FIX":
            if self._inspected and not self._fixed:
                self._fixed = True
                # Strip the '_INVALID' suffix and coerce to numeric
                self._df["salary"] = (
                    self._df["salary"]
                    .astype(str)
                    .str.replace("_INVALID", "", regex=False)
                )
                self._df["salary"] = pd.to_numeric(
                    self._df["salary"], errors="coerce"
                )
                self._issues_found.append("salary column repaired.")
                reward = 0.2
                info["hint"] = "Salary column repaired."
            elif not self._inspected:
                reward = -0.1
                info["hint"] = "Inspect the data before attempting a fix."
            else:
                reward = -0.1
                info["hint"] = "Already fixed."

        elif action == "VALIDATE":
            if self._fixed and not self._validated:
                remaining = self._bad_rows()
                if remaining == 0:
                    self._validated = True
                    reward = 0.2
                    done = True
                    info["hint"] = "Schema is clean. Validation passed."
                else:
                    reward = -0.1
                    info["hint"] = f"{remaining} bad row(s) still remain."
            elif not self._fixed:
                reward = -0.1
                info["hint"] = "Fix the schema before validating."
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
