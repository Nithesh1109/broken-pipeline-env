"""Task 3 (Hard) – Full pipeline debug (null values + type errors + duplicates).

The agent must deal with all three categories of bugs in a single episode:
    1. INSPECT  – detect all issues
    2. FIX      – repair type errors and remove duplicates / nulls
    3. VALIDATE – confirm the dataset passes all quality checks

Correct action sequence (one valid path):
    INSPECT → FIX → FIX → VALIDATE → done
    (two FIX calls needed: first repairs types, second drops nulls/duplicates)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from env.data.generator import generate_base_dataset
from env.data.bug_injector import inject_all_bugs


class Task3Hard:
    """Hard task: detect and fix all pipeline issues in one episode."""

    TASK_ID = "task3"
    DESCRIPTION = (
        "A production pipeline has ingested a corrupted dataset. "
        "The data contains null values, type errors, and duplicate rows. "
        "INSPECT all issues, FIX each category, then VALIDATE the cleaned data."
    )

    # Sub-phases of the FIX step
    _FIX_PHASES = ["type_errors", "null_and_duplicates"]

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        clean = generate_base_dataset(n_rows=40)
        self._df: pd.DataFrame = inject_all_bugs(clean)
        self._actions_taken: List[str] = []
        self._issues_found: List[str] = []
        self._inspected: bool = False
        self._fix_phase: int = 0          # 0 = no fix done, 2 = fully fixed
        self._validated: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _audit(self) -> Dict[str, int]:
        null_count = int(self._df.isna().sum().sum())
        bad_types = int(
            pd.to_numeric(self._df["salary"], errors="coerce").isna().sum()
            - self._df["salary"].isna().sum()
        )
        dup_count = int(self._df.duplicated().sum())
        return {
            "null_values": null_count,
            "type_errors": bad_types,
            "duplicates": dup_count,
        }

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
                audit = self._audit()
                self._issues_found = [
                    f"{audit['null_values']} null value(s) found",
                    f"{audit['type_errors']} type error(s) in salary column",
                    f"{audit['duplicates']} duplicate row(s) found",
                ]
                reward = 0.2
                info["hint"] = f"Audit results: {audit}"
            else:
                reward = -0.1
                info["hint"] = "Already inspected."

        elif action == "FIX":
            if not self._inspected:
                reward = -0.1
                info["hint"] = "Inspect the data before attempting fixes."

            elif self._fix_phase == 0:
                # Phase 1: fix type errors in salary
                self._df["salary"] = (
                    self._df["salary"]
                    .astype(str)
                    .str.replace("_INVALID", "", regex=False)
                )
                self._df["salary"] = pd.to_numeric(
                    self._df["salary"], errors="coerce"
                )
                self._fix_phase = 1
                self._issues_found.append("Type errors in salary fixed.")
                reward = 0.2
                info["hint"] = "Phase 1 complete: type errors repaired."

            elif self._fix_phase == 1:
                # Phase 2: drop nulls and duplicates
                before = len(self._df)
                self._df = self._df.dropna().drop_duplicates().reset_index(drop=True)
                after = len(self._df)
                self._fix_phase = 2
                self._issues_found.append(
                    f"Nulls and duplicates removed ({before - after} rows dropped)."
                )
                reward = 0.2
                info["hint"] = f"Phase 2 complete: {before - after} rows removed."

            else:
                reward = -0.1
                info["hint"] = "All fix phases already applied."

        elif action == "VALIDATE":
            if self._fix_phase < 2:
                reward = -0.1
                info["hint"] = "Complete all FIX phases before validating."
            elif not self._validated:
                audit = self._audit()
                if audit["null_values"] == 0 and audit["duplicates"] == 0:
                    self._validated = True
                    reward = 0.2
                    done = True
                    info["hint"] = "Pipeline fully clean. Validation passed."
                else:
                    reward = -0.1
                    info["hint"] = f"Issues remain: {audit}"
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
