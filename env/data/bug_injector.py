"""env/data/bug_injector.py – Injects deterministic bugs into a DataFrame."""

from __future__ import annotations

import pandas as pd
import numpy as np


def inject_null_values(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Inject null values into 'salary' and 'age' columns."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    null_salary_idx = rng.choice(df.index, size=5, replace=False)
    null_age_idx = rng.choice(df.index, size=3, replace=False)
    df.loc[null_salary_idx, "salary"] = None
    df.loc[null_age_idx, "age"] = None
    return df


def inject_type_errors(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Coerce numeric columns to strings to simulate schema drift."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    bad_idx = rng.choice(df.index, size=4, replace=False)
    # Cast salary to object dtype first so mixed-type assignment is safe
    df["salary"] = df["salary"].astype(object)
    df.loc[bad_idx, "salary"] = (
        df.loc[bad_idx, "salary"].astype(str) + "_INVALID"
    )
    return df


def inject_duplicates(df: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """Duplicate 4 random rows to simulate data-ingestion errors."""
    rng = np.random.default_rng(seed)
    dup_idx = rng.choice(df.index, size=4, replace=False)
    duplicates = df.loc[dup_idx].copy()
    return pd.concat([df, duplicates], ignore_index=True)


def inject_all_bugs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all three bug categories (null, type, duplicate)."""
    df = inject_null_values(df)
    df = inject_type_errors(df)
    df = inject_duplicates(df)
    return df
