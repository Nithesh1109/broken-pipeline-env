"""Bug injector: introduces controlled faults into a clean DataFrame."""

from typing import Optional

import numpy as np
import pandas as pd


def inject_nulls(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    frac: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Replace *frac* fraction of values in *columns* with NaN."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    cols = columns if columns is not None else list(df.columns)
    for col in cols:
        if col == "id":
            continue
        n = max(1, int(len(df) * frac))
        idx = rng.choice(df.index, size=n, replace=False)
        # Cast to object dtype first so NaN can be stored regardless of original dtype
        df[col] = df[col].astype(object)
        df.loc[idx, col] = np.nan
    return df


def inject_wrong_types(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Cast numeric columns to object dtype, simulating type corruption."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    cols = columns if columns is not None else ["age", "score"]
    for col in cols:
        if col not in df.columns:
            continue
        n = max(1, int(len(df) * 0.15))
        idx = rng.choice(df.index, size=n, replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = "INVALID"
    return df


def inject_duplicates(
    df: pd.DataFrame,
    n_duplicates: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Append *n_duplicates* duplicate rows chosen at random."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index, size=n_duplicates, replace=True)
    dup_rows = df.loc[idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


def inject_all_bugs(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Apply nulls, type corruption, and duplicates in sequence."""
    df = inject_nulls(df, seed=seed)
    df = inject_wrong_types(df, seed=seed)
    df = inject_duplicates(df, seed=seed)
    return df
