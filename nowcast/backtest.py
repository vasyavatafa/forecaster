"""Walk-forward, always-refit, one-step-ahead backtest engine.

Protocol (as specified): at every step the model sees the *entire*
history to the left and predicts exactly one point to the right; the
next step adds that one point to history and refits. This produces one
continuous series of hat_x_t forecasts.

Train/test split: rather than being a second, separate procedure, "train"
and "test" are just two slices of that same one-step-ahead forecast
series, split at each column's own 70% mark (`cutoff_frac`) of its
*active* range (from its first to its last observed value) -- so a
column that only starts in 2023 gets its own 70/30 split within its own
history, not the file's global date range. Points before the 70% mark
are still genuine one-step-ahead forecasts (just made with shorter
history / smaller windows) and are reported as "train"; points from the
70% mark to the end are reported as "test", matching the walk-forward
loop the user described (first window = 70%, then +1 point each step).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .data import SeriesCoverage
from .logging_utils import RateLimitedWarner, log_fail, log_ok, progress


@dataclass
class WalkForwardResult:
    column: str
    method_name: str
    params: dict
    forecast: pd.Series  # hat_x_t, index = target date
    actual: pd.Series  # x_t actual, same index
    cutoff_date: pd.Timestamp
    n_train_points: int
    n_test_points: int
    elapsed_seconds: float
    train_metrics: dict
    test_metrics: dict


def _cutoff_date(cov: SeriesCoverage, cutoff_frac: float) -> pd.Timestamp | None:
    if cov.first_valid is None:
        return None
    active = pd.date_range(cov.first_valid, cov.last_valid, freq="7D")
    pos = int(np.floor(cutoff_frac * len(active)))
    pos = min(max(pos, 0), len(active) - 1)
    return active[pos]


def _split_metrics(forecast: pd.Series, actual: pd.Series, cutoff_date: pd.Timestamp, prev_actual: pd.Series):
    train_mask = forecast.index < cutoff_date
    test_mask = forecast.index >= cutoff_date

    def block(mask):
        yt, yp = actual[mask], forecast[mask]
        yprev = prev_actual[mask]
        out = M.compute_basic_metrics(yt, yp)
        out["mase"] = M.mase(yt, yp, actual.dropna())
        out["directional_accuracy"] = M.directional_accuracy(yt, yp, yprev)
        out["n_points"] = int((yt.notna() & yp.notna()).sum())
        return out

    return block(train_mask), block(test_mask), int(train_mask.sum()), int(test_mask.sum())


def run_walk_forward_univariate(
    series: pd.Series,
    coverage: SeriesCoverage,
    method,
    min_history: int = 10,
    cutoff_frac: float = 0.7,
    show_progress: bool = True,
) -> WalkForwardResult | None:
    if coverage.first_valid is None:
        return None
    full_index = series.index
    start_pos = full_index.get_loc(coverage.first_valid) + min_history
    if start_pos >= len(full_index) - 1:
        return None

    method_name = getattr(method, "display_name", type(method).__name__)
    col_name = str(series.name)
    stage = f"{method_name} | {col_name}"
    warner = RateLimitedWarner(stage)

    fc_vals, dates = [], []
    t0 = time.perf_counter()
    steps = range(start_pos, len(full_index) - 1)
    for t_pos in progress(steps, desc=stage, leave=False, disable=not show_progress):
        history = series.iloc[: t_pos + 1]
        target_date = full_index[t_pos + 1]
        try:
            pred = method.step(history)
        except Exception as exc:  # a single bad window shouldn't kill the whole run
            warner.warn(str(target_date.date()), exc)
            pred = np.nan
        fc_vals.append(pred)
        dates.append(target_date)
    elapsed = time.perf_counter() - t0

    forecast = pd.Series(fc_vals, index=pd.DatetimeIndex(dates))
    actual = series.reindex(forecast.index)
    prev_actual = series.shift(1).reindex(forecast.index)

    cutoff = _cutoff_date(coverage, cutoff_frac)
    if cutoff is None or cutoff < forecast.index.min():
        cutoff = forecast.index.min()
    train_m, test_m, n_tr, n_te = _split_metrics(forecast, actual, cutoff, prev_actual)

    step_warn = warner.summary()
    if forecast.notna().sum() == 0 and warner.count > 0:
        # every single step raised -- this is a stage failure (bad config,
        # not just an isolated bad window), not a success with some noise
        log_fail(stage, f"all {warner.count} step(s) failed, see [WARN] lines above")
    elif show_progress:
        detail = f"n_train={n_tr} n_test={n_te} test_mase={test_m.get('mase'):.4g} ({elapsed:.2f}s)"
        if step_warn:
            detail += f" [{step_warn}]"
        log_ok(stage, detail)

    return WalkForwardResult(
        column=col_name,
        method_name=method_name,
        params=getattr(method, "__dict__", {}),
        forecast=forecast,
        actual=actual,
        cutoff_date=cutoff,
        n_train_points=n_tr,
        n_test_points=n_te,
        elapsed_seconds=elapsed,
        train_metrics=train_m,
        test_metrics=test_m,
    )


def run_walk_forward_multivariate(
    df: pd.DataFrame,
    coverage: dict[str, SeriesCoverage],
    method,
    min_history: int = 10,
    cutoff_frac: float = 0.7,
    show_progress: bool = True,
) -> dict[str, WalkForwardResult]:
    """One shared walk-forward loop producing forecasts for all columns
    of `df` jointly at every step; metrics are then split per column
    using each column's own cutoff date."""
    full_index = df.index
    first_any = min(c.first_valid for c in coverage.values() if c.first_valid is not None)
    start_pos = full_index.get_loc(first_any) + min_history
    if start_pos >= len(full_index) - 1:
        return {}

    method_name = getattr(method, "display_name", type(method).__name__)
    stage = f"{method_name} | {'+'.join(df.columns)}"
    warner = RateLimitedWarner(stage)

    rows, dates = [], []
    t0 = time.perf_counter()
    steps = range(start_pos, len(full_index) - 1)
    for t_pos in progress(steps, desc=stage, leave=False, disable=not show_progress):
        history = df.iloc[: t_pos + 1]
        target_date = full_index[t_pos + 1]
        try:
            pred_row = method.step(history)
        except Exception as exc:  # a single bad window shouldn't kill the whole run
            warner.warn(str(target_date.date()), exc)
            pred_row = pd.Series(np.nan, index=df.columns)
        rows.append(pred_row)
        dates.append(target_date)
    elapsed = time.perf_counter() - t0

    forecast_df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    results: dict[str, WalkForwardResult] = {}
    step_warn = warner.summary()

    for col in df.columns:
        cov = coverage[col]
        col_stage = f"{method_name} | {col}"
        if cov.first_valid is None or col not in forecast_df.columns:
            continue
        forecast = forecast_df[col]
        actual = df[col].reindex(forecast.index)
        prev_actual = df[col].shift(1).reindex(forecast.index)
        cutoff = _cutoff_date(cov, cutoff_frac)
        if cutoff is None or cutoff < forecast.index.min():
            cutoff = forecast.index.min()
        train_m, test_m, n_tr, n_te = _split_metrics(forecast, actual, cutoff, prev_actual)

        if forecast.notna().sum() == 0 and warner.count > 0:
            # every single step raised for this joint model -- a stage
            # failure (bad config), not a success with some noise
            log_fail(col_stage, f"all {warner.count} step(s) failed, see [WARN] lines above")
        elif show_progress:
            detail = f"n_train={n_tr} n_test={n_te} test_mase={test_m.get('mase'):.4g}"
            if step_warn:
                detail += f" [{step_warn}]"
            log_ok(col_stage, detail)

        results[col] = WalkForwardResult(
            column=col,
            method_name=method_name,
            params=getattr(method, "__dict__", {}),
            forecast=forecast,
            actual=actual,
            cutoff_date=cutoff,
            n_train_points=n_tr,
            n_test_points=n_te,
            elapsed_seconds=elapsed * (n_tr + n_te) / max(len(dates), 1),
            train_metrics=train_m,
            test_metrics=test_m,
        )
    return results
