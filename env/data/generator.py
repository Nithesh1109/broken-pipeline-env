"""Synthetic dataset generator using pandas."""

from typing import Dict, Any

import numpy as np
import pandas as pd


_RNG_SEED = 42


def generate_dataset(n_rows: int = 50, seed: int = _RNG_SEED) -> pd.DataFrame:
    """Return a clean synthetic tabular dataset.

    Columns
    -------
    id          : int   – unique row identifier
    name        : str   – sample name string
    age         : int   – integer age (18–80)
    score       : float – numeric score (0.0–100.0)
    category    : str   – one of A / B / C
    is_active   : bool  – boolean flag
    """
    rng = np.random.default_rng(seed)

    categories = rng.choice(["A", "B", "C"], size=n_rows)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "name": [f"user_{i:03d}" for i in range(1, n_rows + 1)],
            "age": rng.integers(18, 81, size=n_rows),
            "score": np.round(rng.uniform(0.0, 100.0, size=n_rows), 2),
            "category": categories,
            "is_active": rng.choice([True, False], size=n_rows),
        }
    )
    return df


def dataset_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a DataFrame to a JSON-serialisable snapshot dict."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": df.head(5).to_dict(orient="records"),
    }
