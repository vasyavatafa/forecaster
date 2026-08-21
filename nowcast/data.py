"""Loading and light validation of weekly multivariate series from xlsx.

Handles the two quirks the user flagged:
  * columns that start at different dates (e.g. main block from 2020,
    others joining in 2022/2023/2024) -> tracked as `first_valid`/`last_valid`
    per column, nothing is silently backfilled before a series' true start.
  * missing values inside an otherwise active column -> left as NaN; each
    forecasting method decides how to handle NaNs in its own window
    (see methods.py), this module only reports on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SeriesCoverage:
    column: str
    first_valid: pd.Timestamp | None
    last_valid: pd.Timestamp | None
    n_obs: int
    n_expected: int
    n_missing_interior: int  # NaNs strictly between first_valid and last_valid
    coverage_ratio: float  # n_obs / n_expected over [first_valid, last_valid]


@dataclass
class LoadedData:
    df: pd.DataFrame  # index = weekly DatetimeIndex, columns = series
    date_col: str
    coverage: dict[str, SeriesCoverage] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for col, cov in self.coverage.items():
            rows.append(
                {
                    "column": col,
                    "first_valid": cov.first_valid,
                    "last_valid": cov.last_valid,
                    "n_obs": cov.n_obs,
                    "n_expected": cov.n_expected,
                    "n_missing_interior": cov.n_missing_interior,
                    "coverage_ratio": round(cov.coverage_ratio, 4),
                }
            )
        return pd.DataFrame(rows).set_index("column")


def _infer_weekly_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Build the expected complete weekly grid spanning the data, anchored
    on the most common weekday actually observed (handles data that isn't
    exactly Mon/Fri-stamped)."""
    if len(index) == 0:
        return index
    weekday_counts = pd.Series(index.weekday).value_counts()
    anchor_weekday = int(weekday_counts.idxmax())
    start, end = index.min(), index.max()
    # shift start to the anchor weekday
    shift = (anchor_weekday - start.weekday()) % 7
    start = start + pd.Timedelta(days=shift)
    return pd.date_range(start=start, end=end, freq="7D")


def load_weekly_xlsx(
    path: str,
    date_col: str = "date",
    columns: list[str] | None = None,
    sheet_name: int | str = 0,
) -> LoadedData:
    """Load a weekly xlsx file into a regular weekly-indexed DataFrame.

    Parameters
    ----------
    path : path to .xlsx
    date_col : name of the date column
    columns : optional subset of value columns to keep (default: all
        columns except date_col)
    sheet_name : sheet to read (default first sheet)
    """
    raw = pd.read_excel(path, sheet_name=sheet_name)
    if date_col not in raw.columns:
        raise ValueError(f"date_col='{date_col}' not found in columns: {list(raw.columns)}")

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.sort_values(date_col)
    dupes = raw[date_col].duplicated().sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate timestamps in '{date_col}' after parsing; "
            "dedupe the source file before loading."
        )
    raw = raw.set_index(date_col)

    if columns is None:
        columns = [c for c in raw.columns if c != date_col]
    else:
        missing = [c for c in columns if c not in raw.columns]
        if missing:
            raise ValueError(f"requested columns not in file: {missing}")

    df = raw[columns].apply(pd.to_numeric, errors="coerce")

    full_grid = _infer_weekly_grid(df.index)
    df = df.reindex(full_grid)
    df.index.name = date_col

    coverage = compute_coverage(df)

    return LoadedData(df=df, date_col=date_col, coverage=coverage)


def compute_coverage(df: pd.DataFrame) -> dict[str, SeriesCoverage]:
    """Per-column first/last valid date, expected vs. observed count and
    interior-missing count, computed on whatever range `df` currently
    covers (call again after slicing by start_train/end_train)."""
    coverage: dict[str, SeriesCoverage] = {}
    for col in df.columns:
        s = df[col]
        valid_idx = s.index[s.notna()]
        if len(valid_idx) == 0:
            coverage[col] = SeriesCoverage(col, None, None, 0, 0, 0, 0.0)
            continue
        first_valid, last_valid = valid_idx.min(), valid_idx.max()
        window = s.loc[first_valid:last_valid]
        n_expected = len(window)
        n_obs = int(window.notna().sum())
        n_missing_interior = n_expected - n_obs
        coverage[col] = SeriesCoverage(
            column=col,
            first_valid=first_valid,
            last_valid=last_valid,
            n_obs=n_obs,
            n_expected=n_expected,
            n_missing_interior=n_missing_interior,
            coverage_ratio=n_obs / n_expected if n_expected else 0.0,
        )
    return coverage


def slice_train_range(
    df: pd.DataFrame,
    start_train: str | pd.Timestamp | None,
    end_train: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Bound the overall working range. Does NOT touch per-column
    first/last valid dates -- a column that only starts in 2023 will
    still be all-NaN before 2023 even if start_train is earlier."""
    out = df
    if start_train is not None:
        out = out.loc[pd.Timestamp(start_train):]
    if end_train is not None:
        out = out.loc[:pd.Timestamp(end_train)]
    return out
