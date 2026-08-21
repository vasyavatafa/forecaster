"""End-to-end demo on the synthetic dataset:
  1. generate synthetic weekly data (staggered starts, gaps, 3 shock episodes)
  2. run all 5 methods with default hyperparameters -> train/test metrics table
  3. estimate grid-search runtime, then run the real grid search, top-10 per method
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from nowcast.pipeline import NowcastPipeline, DEFAULT_PARAM_GRIDS
from synthetic import save_synthetic_xlsx

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

DATA_PATH = "data/synthetic_weekly.xlsx"
RESULTS_DIR = "results"


def main():
    print("1) generating synthetic data ...")
    save_synthetic_xlsx(DATA_PATH)

    pipe = NowcastPipeline(DATA_PATH, date_col="date")
    print("\ncoverage:")
    print(pipe.coverage_summary())

    print("\n2) running all 5 methods with default hyperparameters ...")
    t0 = time.perf_counter()
    table = pipe.run_all_methods(save_path=f"{RESULTS_DIR}/all_methods_summary.csv")
    print(f"done in {time.perf_counter() - t0:.1f}s")
    cols = ["method", "column", "n_train", "n_test", "train_rmse", "test_rmse",
            "test_smape", "test_r2", "test_mase", "test_directional_accuracy"]
    print(table[cols].to_string(index=False))

    print("\n3) estimating grid-search runtime (default grids, all columns) ...")
    est = pipe.estimate_all_methods(sample_configs=3)
    print(est)

    print("\n4) running the real grid search, top-10 per method ...")
    t0 = time.perf_counter()
    out = pipe.search_all_methods(top_k=10, n_jobs=1, save_dir=RESULTS_DIR)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    for method, (all_df, top) in out.items():
        print(f"\n--- {method} ---")
        if isinstance(top, dict):
            for col, t in top.items():
                print(col)
                print(t[["params", "test_rmse", "test_smape", "test_r2"]].head(5).to_string(index=False))
        else:
            print(top[["params", "mean_test_rmse"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
