"""env/data/generator.py – Synthetic dataset factory."""

from __future__ import annotations

import pandas as pd
import numpy as np


# Fixed seed so datasets are deterministic.
_RNG = np.random.default_rng(42)


def generate_base_dataset(n_rows: int = 50) -> pd.DataFrame:
    """Return a clean synthetic employee-sales dataset."""
    rng = np.random.default_rng(42)

    employee_ids = [f"EMP{i:03d}" for i in range(1, n_rows + 1)]
    departments = rng.choice(["Sales", "Marketing", "Engineering", "HR"], size=n_rows)
    ages = rng.integers(22, 60, size=n_rows).tolist()
    salaries = rng.integers(40_000, 120_000, size=n_rows).tolist()
    sales = rng.integers(0, 500, size=n_rows).tolist()

    return pd.DataFrame(
        {
            "employee_id": employee_ids,
            "department": departments,
            "age": ages,
            "salary": salaries,
            "sales_count": sales,
        }
    )
