"""Hyperparameter grid search across all 5 method families, with a
runtime estimator (time a handful of configs, extrapolate to the full
grid) and a top-10-per-method-family selector.
"""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .backtest import run_walk_forward_multivariate, run_walk_forward_univariate
from .data import SeriesCoverage
from .methods import METHOD_REGISTRY, MULTIVARIATE_METHODS


def expand_grid(param_grid: dict) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def _run_one_univariate(method_name, params, series, coverage, min_history, cutoff_frac):
    method = METHOD_REGISTRY[method_name](**params)
    res = run_walk_forward_univariate(series, coverage, method, min_history, cutoff_frac)
    if res is None:
        return None
    row = {"method": method_name, "column": res.column, "params": params,
           "elapsed_seconds": res.elapsed_seconds, "cutoff_date": res.cutoff_date,
           "n_train": res.n_train_points, "n_test": res.n_test_points}
    row.update({f"train_{k}": v for k, v in res.train_metrics.items()})
    row.update({f"test_{k}": v for k, v in res.test_metrics.items()})
    return row


def _run_one_multivariate(method_name, params, df, coverage, min_history, cutoff_frac):
    method = METHOD_REGISTRY[method_name](**params)
    results = run_walk_forward_multivariate(df, coverage, method, min_history, cutoff_frac)
    rows = []
    for col, res in results.items():
        row = {"method": method_name, "column": col, "params": params,
               "elapsed_seconds": res.elapsed_seconds, "cutoff_date": res.cutoff_date,
               "n_train": res.n_train_points, "n_test": res.n_test_points}
        row.update({f"train_{k}": v for k, v in res.train_metrics.items()})
        row.update({f"test_{k}": v for k, v in res.test_metrics.items()})
        rows.append(row)
    return rows


@dataclass
class RuntimeEstimate:
    method: str
    n_configs: int
    n_columns: int
    sampled_configs: int
    avg_seconds_per_config_per_target: float
    estimated_total_seconds: float

    def __str__(self):
        mins = self.estimated_total_seconds / 60
        return (f"[{self.method}] {self.n_configs} configs x {self.n_columns} column(s) "
                f"~ {self.avg_seconds_per_config_per_target:.3f}s/config -> "
                f"estimated total {self.estimated_total_seconds:.1f}s (~{mins:.1f} min)")


def estimate_runtime(
    method_name: str,
    param_grid: dict,
    df: pd.DataFrame,
    coverage: dict[str, SeriesCoverage],
    columns: list[str],
    min_history: int = 10,
    cutoff_frac: float = 0.7,
    sample_configs: int = 3,
    seed: int = 0,
) -> RuntimeEstimate:
    combos = expand_grid(param_grid)
    rng = random.Random(seed)
    sample = rng.sample(combos, min(sample_configs, len(combos)))
    is_multi = method_name in MULTIVARIATE_METHODS

    times = []
    for cfg in sample:
        t0 = time.perf_counter()
        if is_multi:
            _run_one_multivariate(method_name, cfg, df[columns], {c: coverage[c] for c in columns}, min_history, cutoff_frac)
        else:
            col = columns[0]
            _run_one_univariate(method_name, cfg, df[col], coverage[col], min_history, cutoff_frac)
        times.append(time.perf_counter() - t0)
    avg = float(np.mean(times))

    if is_multi:
        total = avg * len(combos)
    else:
        total = avg * len(combos) * len(columns)

    return RuntimeEstimate(
        method=method_name,
        n_configs=len(combos),
        n_columns=len(columns),
        sampled_configs=len(sample),
        avg_seconds_per_config_per_target=avg,
        estimated_total_seconds=total,
    )


def run_grid_search(
    method_name: str,
    param_grid: dict,
    df: pd.DataFrame,
    coverage: dict[str, SeriesCoverage],
    columns: list[str],
    min_history: int = 10,
    cutoff_frac: float = 0.7,
    n_jobs: int = 1,
    rank_metric: str = "test_rmse",
    top_k: int = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | pd.DataFrame]:
    """Runs the full grid for one method family.

    Returns (all_results_df, top_k) where for univariate methods top_k
    is a dict {column: DataFrame of top_k rows}, and for multivariate
    methods it is a single DataFrame (one config per row, ranked by the
    mean of `rank_metric` across `columns`).
    """
    combos = expand_grid(param_grid)
    is_multi = method_name in MULTIVARIATE_METHODS

    if is_multi:
        sub_cov = {c: coverage[c] for c in columns}
        jobs = (
            delayed(_run_one_multivariate)(method_name, cfg, df[columns], sub_cov, min_history, cutoff_frac)
            for cfg in combos
        )
        nested = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)
        rows = [r for sub in nested for r in sub]
    else:
        jobs = (
            delayed(_run_one_univariate)(method_name, cfg, df[col], coverage[col], min_history, cutoff_frac)
            for cfg in combos
            for col in columns
        )
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)
        rows = [r for r in rows if r is not None]

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df, ({} if not is_multi else all_df)

    all_df["params_str"] = all_df["params"].apply(lambda d: str(sorted(d.items())))

    if is_multi:
        agg = (
            all_df.groupby("params_str")[rank_metric]
            .mean()
            .rename("mean_" + rank_metric)
            .reset_index()
        )
        first_params = all_df.drop_duplicates("params_str")[["params_str", "params"]]
        agg = agg.merge(first_params, on="params_str")
        agg = agg.sort_values("mean_" + rank_metric, ascending=True).head(top_k)
        top = agg.drop(columns=["params_str"]).reset_index(drop=True)
        return all_df, top
    else:
        top_by_col: dict[str, pd.DataFrame] = {}
        for col, g in all_df.groupby("column"):
            g_sorted = g.sort_values(rank_metric, ascending=True).head(top_k)
            top_by_col[col] = g_sorted.reset_index(drop=True)
        return all_df, top_by_col
