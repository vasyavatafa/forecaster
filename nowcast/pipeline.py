"""Main user-facing entry point.

    from nowcast.pipeline import NowcastPipeline

    pipe = NowcastPipeline(
        path="data/weekly.xlsx",
        date_col="date",
        columns=None,                 # None -> all non-date columns
        method="bayesian_ar",         # one of METHOD_REGISTRY keys
        start_train="2022-01-20",
        end_train=None,
    )
    results = pipe.run()              # {column: WalkForwardResult}
    table = pipe.run_all_methods()    # one row per (method, column), train+test metrics
    wide  = pipe.forecast_table(list_features=[...], list_methods=[...])
                                       # input df + one "{col}_{method}_frcst" column per requested pair
    est = pipe.estimate_runtime("nn_revin", param_grid={...})
    all_df, top10 = pipe.grid_search("rolling_mean", param_grid={...}, top_k=10)

Per-method default hyperparameters are read from config/default_params.yaml
(see that file for what every field controls); method_params=... passed at
call time always overrides those file defaults for that call only.

Every stage (one walk-forward run for a given method+column, one method
in run_all_methods/forecast_table) is progress-tracked with tqdm and logs
a colored [OK]/[FAIL] line when it finishes; a failure on one column/
method is caught and logged, and the pipeline moves on to the next one
instead of stopping the whole run (see nowcast/logging_utils.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import WalkForwardResult, run_walk_forward_multivariate, run_walk_forward_univariate
from .config import load_default_params
from .data import LoadedData, compute_coverage, load_weekly_xlsx, slice_train_range
from .logging_utils import log_fail, log_info, progress
from .methods import METHOD_REGISTRY, MULTIVARIATE_METHODS
from .search import RuntimeEstimate, estimate_runtime, run_grid_search


DEFAULT_PARAM_GRIDS = {name: cls.param_grid for name, cls in METHOD_REGISTRY.items()}


class NowcastPipeline:
    def __init__(
        self,
        path: str,
        date_col: str = "date",
        columns: list[str] | None = None,
        method: str = "rolling_mean",
        start_train: str | pd.Timestamp | None = None,
        end_train: str | pd.Timestamp | None = None,
        cutoff_frac: float = 0.7,
        min_history: int = 10,
        method_params: dict | None = None,
        sheet_name: int | str = 0,
        config_path: str | None = None,
    ):
        if method not in METHOD_REGISTRY:
            raise ValueError(f"unknown method '{method}', choose from {list(METHOD_REGISTRY)}")

        self.loaded: LoadedData = load_weekly_xlsx(path, date_col=date_col, columns=columns, sheet_name=sheet_name)
        self.date_col = date_col
        self.df = slice_train_range(self.loaded.df, start_train, end_train)
        self.coverage = compute_coverage(self.df)
        self.columns = list(self.df.columns)
        self.method = method
        self.cutoff_frac = cutoff_frac
        self.min_history = min_history

        # single-value defaults per method, from config/default_params.yaml;
        # a method_params={...} override passed to a specific call always
        # wins over these
        self.yaml_defaults = load_default_params(config_path)
        self.method_params = method_params or {}

    # ------------------------------------------------------------------
    def _resolve_params(self, method: str, override: dict | None) -> dict:
        base = dict(self.yaml_defaults.get(method, {}))
        if override:
            base.update(override)
        elif method == self.method and self.method_params:
            base.update(self.method_params)
        return base

    # ------------------------------------------------------------------
    def coverage_summary(self) -> pd.DataFrame:
        """Coverage over the working range (after start_train/end_train
        slicing). Use `self.loaded.summary()` for coverage over the raw
        file before any slicing."""
        return LoadedData(df=self.df, date_col=self.date_col, coverage=self.coverage).summary()

    def set_columns(self, columns: list[str]):
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"columns not available: {missing}")
        self.columns = columns

    # ------------------------------------------------------------------
    def run(
        self,
        columns: list[str] | None = None,
        method: str | None = None,
        method_params: dict | None = None,
        show_progress: bool = True,
    ) -> dict[str, WalkForwardResult]:
        columns = columns or self.columns
        method = method or self.method
        params = self._resolve_params(method, method_params)
        method_cls = METHOD_REGISTRY[method]

        if method in MULTIVARIATE_METHODS:
            try:
                m = method_cls(**params)
                sub_cov = {c: self.coverage[c] for c in columns}
                return run_walk_forward_multivariate(
                    self.df[columns], sub_cov, m, self.min_history, self.cutoff_frac, show_progress
                )
            except Exception as exc:
                log_fail(f"{method} | {'+'.join(columns)}", exc)
                return {}
        else:
            out = {}
            for col in progress(columns, desc=f"{method} (columns)", leave=False, disable=not show_progress):
                try:
                    m = method_cls(**params)
                    res = run_walk_forward_univariate(
                        self.df[col], self.coverage[col], m, self.min_history, self.cutoff_frac, show_progress
                    )
                    if res is not None:
                        out[col] = res
                    else:
                        log_fail(f"{method} | {col}", "not enough history to forecast even one point -- skipped")
                except Exception as exc:
                    # highlight the failure for this feature and move on to the next one
                    log_fail(f"{method} | {col}", exc)
                    continue
            return out

    # ------------------------------------------------------------------
    def run_all_methods(
        self,
        columns: list[str] | None = None,
        method_params: dict[str, dict] | None = None,
        methods: list[str] | None = None,
        save_path: str | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Runs every method (with its default hyperparameters, or the
        override given per-method in `method_params`) on every column,
        returns one row per (method, column) with train + test metrics."""
        columns = columns or self.columns
        methods = methods or list(METHOD_REGISTRY.keys())
        method_params = method_params or {}

        rows = []
        for method in progress(methods, desc="run_all_methods", disable=not show_progress):
            params = method_params.get(method, {})
            results = self.run(columns=columns, method=method, method_params=params, show_progress=show_progress)
            for col, res in results.items():
                row = {
                    "method": method,
                    "column": col,
                    "cutoff_date": res.cutoff_date,
                    "n_train": res.n_train_points,
                    "n_test": res.n_test_points,
                    "elapsed_seconds": res.elapsed_seconds,
                }
                row.update({f"train_{k}": v for k, v in res.train_metrics.items()})
                row.update({f"test_{k}": v for k, v in res.test_metrics.items()})
                rows.append(row)
        table = pd.DataFrame(rows)
        if save_path:
            table.to_csv(save_path, index=False)
        return table

    # ------------------------------------------------------------------
    def forecast_table(
        self,
        list_features: list[str] | None = None,
        list_methods: list[str] | None = None,
        method_params: dict[str, dict] | None = None,
        keep_original: bool = True,
        save_path: str | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Returns a dataframe on the exact same (weekly) index as the
        input file -- with the original columns kept (`keep_original`) --
        plus one new column per requested (feature, method) pair, named
        "{feature}_{method}_frcst", holding that method's walk-forward
        hat_x_t. Missing where the method hasn't got enough history yet
        (or a stage failed -- see the [FAIL] log line for why).
        """
        features = list_features or self.columns
        methods = list_methods or list(METHOD_REGISTRY.keys())
        method_params = method_params or {}

        base = self.df.copy() if keep_original else pd.DataFrame(index=self.df.index)

        # collect every forecast column into a plain dict first and add them
        # all to the output in one pd.concat at the end, instead of
        # `out[colname] = ...` inside the loop -- assigning columns one at a
        # time onto a DataFrame that keeps growing (n_features x n_methods
        # times) is exactly the pattern pandas warns about ("DataFrame is
        # highly fragmented") and gets slower with every column added.
        forecast_cols: dict[str, pd.Series] = {}
        for method in progress(methods, desc="forecast_table", disable=not show_progress):
            params = method_params.get(method, {})
            results = self.run(columns=features, method=method, method_params=params, show_progress=show_progress)
            for col in features:
                colname = f"{col}_{method}_frcst"
                res = results.get(col)
                forecast_cols[colname] = (
                    res.forecast.reindex(base.index) if res is not None
                    else pd.Series(np.nan, index=base.index)
                )

        forecasts = pd.DataFrame(forecast_cols, index=base.index)
        out = pd.concat([base, forecasts], axis=1)

        if save_path:
            out.to_csv(save_path)
        log_info(f"forecast_table done: {len(features)} feature(s) x {len(methods)} method(s) "
                  f"= {len(features) * len(methods)} forecast column(s)")
        return out

    # ------------------------------------------------------------------
    def estimate_runtime(
        self,
        method: str,
        param_grid: dict | None = None,
        columns: list[str] | None = None,
        sample_configs: int = 3,
    ) -> RuntimeEstimate:
        columns = columns or self.columns
        param_grid = param_grid or DEFAULT_PARAM_GRIDS[method]
        return estimate_runtime(
            method, param_grid, self.df, self.coverage, columns,
            self.min_history, self.cutoff_frac, sample_configs,
        )

    def estimate_all_methods(
        self,
        param_grids: dict[str, dict] | None = None,
        columns: list[str] | None = None,
        sample_configs: int = 3,
    ) -> pd.DataFrame:
        columns = columns or self.columns
        param_grids = param_grids or DEFAULT_PARAM_GRIDS
        rows = []
        for method, grid in param_grids.items():
            est = self.estimate_runtime(method, grid, columns, sample_configs)
            rows.append({
                "method": est.method, "n_configs": est.n_configs, "n_columns": est.n_columns,
                "avg_sec_per_config": round(est.avg_seconds_per_config_per_target, 4),
                "estimated_total_seconds": round(est.estimated_total_seconds, 1),
                "estimated_total_minutes": round(est.estimated_total_seconds / 60, 2),
            })
        df = pd.DataFrame(rows).sort_values("estimated_total_seconds", ascending=False)
        df.loc["TOTAL", ["estimated_total_seconds", "estimated_total_minutes"]] = [
            df["estimated_total_seconds"].sum(), df["estimated_total_minutes"].sum()
        ]
        return df

    # ------------------------------------------------------------------
    def grid_search(
        self,
        method: str,
        param_grid: dict | None = None,
        columns: list[str] | None = None,
        top_k: int = 10,
        n_jobs: int = 1,
        rank_metric: str = "test_rmse",
        show_progress: bool = True,
    ):
        columns = columns or self.columns
        param_grid = param_grid or DEFAULT_PARAM_GRIDS[method]
        return run_grid_search(
            method, param_grid, self.df, self.coverage, columns,
            self.min_history, self.cutoff_frac, n_jobs, rank_metric, top_k, show_progress,
        )

    def search_all_methods(
        self,
        param_grids: dict[str, dict] | None = None,
        columns: list[str] | None = None,
        top_k: int = 10,
        n_jobs: int = 1,
        rank_metric: str = "test_rmse",
        save_dir: str | None = None,
        show_progress: bool = True,
    ) -> dict[str, tuple[pd.DataFrame, object]]:
        """Runs the full grid search for all 5 method families, returns
        {method: (all_results_df, top_k)}. If save_dir is given, writes
        one CSV of all results and one of top-k per method."""
        columns = columns or self.columns
        param_grids = param_grids or DEFAULT_PARAM_GRIDS
        out = {}
        for method, grid in progress(param_grids.items(), desc="search_all_methods",
                                      total=len(param_grids), disable=not show_progress):
            try:
                all_df, top = self.grid_search(method, grid, columns, top_k, n_jobs, rank_metric, show_progress)
            except Exception as exc:
                log_fail(f"grid_search | {method}", exc)
                continue
            out[method] = (all_df, top)
            if save_dir:
                all_df.to_csv(f"{save_dir}/{method}_all_results.csv", index=False)
                if isinstance(top, dict):
                    for col, t in top.items():
                        t.to_csv(f"{save_dir}/{method}_top{top_k}_{col}.csv", index=False)
                else:
                    top.to_csv(f"{save_dir}/{method}_top{top_k}.csv", index=False)
        return out
