"""Synthetic weekly multivariate dataset for testing the nowcast pipeline.

Mimics the real data's quirks:
  * weekly cadence from 2020-01-06
  * 5 differently-shaped latent factors drive correlated co-movement
    across series (with different loadings/lags per series), so
    VAR/NN methods have real -- and structurally varied -- cross-series
    correlation to exploit:
      f1: AR(1)
      f2: AR(2) + linear trend(t)
      f3: AR(2) + doubled innovation volatility
      f4: AR(3) + quadratic trend(t^2)
      f5: AR(1) + 0.02*sin(t)
  * staggered start dates: core block from 2020, one series joining in
    2022, one in 2023, one in 2024
  * missing values: scattered single-week gaps + one longer outage block
  * 2-3 noisy/shock episodes (level jump + variance spike for a few
    weeks), simulating real economic-event episodes
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ar_process(n: int, phis: list[float], std: float, rng: np.random.Generator) -> np.ndarray:
    """AR(p) with phis=[phi_1,...,phi_p] (lag-1 coefficient first),
    Gaussian innovations of std `std`. Missing lags during burn-in
    (i < p) are treated as 0."""
    x = np.zeros(n)
    p = len(phis)
    for i in range(n):
        val = 0.0
        for k in range(p):
            if i - 1 - k >= 0:
                val += phis[k] * x[i - 1 - k]
        x[i] = val + rng.normal(0, std)
    return x


def generate_synthetic_dataset(
    start: str = "2020-01-06",
    end: str = "2026-08-14",
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="7D")
    n = len(dates)
    t = np.arange(n)

    # ---- 5 latent factors, each a different process shape ----
    baseline_std = 1.0
    f1 = _ar_process(n, [0.9], baseline_std, rng)                                 # AR(1)
    f2 = _ar_process(n, [0.6, 0.25], baseline_std, rng) + 0.01 * t                 # AR(2) + trend(t)
    f3 = _ar_process(n, [0.5, 0.3], baseline_std * 2.0, rng)                       # AR(2) + volatility*2
    f4 = _ar_process(n, [0.5, 0.2, 0.15], baseline_std, rng) + 0.00004 * t ** 2    # AR(3) + trend(t^2)
    f5 = _ar_process(n, [0.9], baseline_std, rng) + 0.02 * np.sin(t)               # AR(1) + 0.02*sin(t)

    def build_series(factor, loading, lag, own_ar, trend_per_week, seasonal_amp, noise_std, base):
        f_lag = np.concatenate([np.zeros(lag), factor[: n - lag]]) if lag > 0 else factor.copy()
        idio = np.zeros(n)
        for i in range(1, n):
            idio[i] = own_ar * idio[i - 1] + rng.normal(0, noise_std)
        seasonal = seasonal_amp * np.sin(2 * np.pi * t / 52.0)
        trend = trend_per_week * t
        return base + trend + seasonal + loading * f_lag + idio

    # each series is driven by one factor type (new_activity_index blends
    # two, to also exercise multi-factor loadings)
    gdp_proxy = build_series(f1, loading=1.0, lag=0, own_ar=0.3, trend_per_week=0.05, seasonal_amp=1.5, noise_std=0.6, base=100.0)
    inflation_proxy = build_series(f2, loading=0.6, lag=2, own_ar=0.5, trend_per_week=0.02, seasonal_amp=0.5, noise_std=0.4, base=3.0)
    rates_proxy = build_series(f3, loading=0.4, lag=4, own_ar=0.7, trend_per_week=0.01, seasonal_amp=0.2, noise_std=0.15, base=2.0)
    consumer_sentiment = build_series(f4, loading=0.8, lag=1, own_ar=0.4, trend_per_week=-0.01, seasonal_amp=2.0, noise_std=1.0, base=50.0)
    credit_spread = build_series(f5, loading=-0.5, lag=3, own_ar=0.6, trend_per_week=-0.005, seasonal_amp=0.1, noise_std=0.2, base=1.5)
    new_activity_index = build_series(0.6 * f1 + 0.4 * f3, loading=0.3, lag=0, own_ar=0.5, trend_per_week=0.03, seasonal_amp=1.0, noise_std=0.7, base=20.0)

    df = pd.DataFrame(
        {
            "gdp_proxy": gdp_proxy,
            "inflation_proxy": inflation_proxy,
            "rates_proxy": rates_proxy,
            "consumer_sentiment": consumer_sentiment,
            "credit_spread": credit_spread,
            "new_activity_index": new_activity_index,
        },
        index=dates,
    )

    # ---- shock episodes: level jump + variance spike, 2-3 short windows ----
    shock_windows = [
        ("2020-03-06", "2020-05-01"),   # covid-like shock, hits everything
        ("2022-09-01", "2022-10-15"),   # mid-sample shock
        ("2024-11-01", "2024-12-15"),   # late-sample shock
    ]
    for start_s, end_s in shock_windows:
        mask = (df.index >= start_s) & (df.index <= end_s)
        k = mask.sum()
        if k == 0:
            continue
        jump = rng.normal(0, 4.0)
        for col in df.columns:
            extra_noise = rng.normal(0, df[col].std() * 0.8, size=k)
            df.loc[mask, col] = df.loc[mask, col] + jump * rng.uniform(0.5, 1.2) + extra_noise

    # ---- staggered start dates: blank out the beginning of some columns ----
    df.loc[df.index < "2022-01-07", "consumer_sentiment"] = np.nan
    df.loc[df.index < "2023-01-06", "credit_spread"] = np.nan
    df.loc[df.index < "2024-01-05", "new_activity_index"] = np.nan

    # ---- scattered single-week gaps in otherwise-active columns ----
    for col in ["gdp_proxy", "inflation_proxy", "rates_proxy"]:
        valid_idx = df.index[df[col].notna()]
        n_gaps = max(1, int(0.02 * len(valid_idx)))
        gap_dates = rng.choice(valid_idx, size=n_gaps, replace=False)
        df.loc[gap_dates, col] = np.nan

    # one longer outage block simulating a data-provider gap
    df.loc["2021-06-07":"2021-07-05", "inflation_proxy"] = np.nan
    consumer_active = df.index[df["consumer_sentiment"].notna()]
    if len(consumer_active) > 20:
        block_start = consumer_active[10]
        block_end = consumer_active[14]
        df.loc[block_start:block_end, "consumer_sentiment"] = np.nan

    df.index.name = "date"
    return df


def save_synthetic_xlsx(path: str, **kwargs) -> pd.DataFrame:
    df = generate_synthetic_dataset(**kwargs)
    out = df.reset_index()
    out.to_excel(path, index=False, sheet_name="weekly")
    return df


if __name__ == "__main__":
    df = save_synthetic_xlsx("data/synthetic_weekly.xlsx")
    print(df.describe())
    print(df.isna().sum())
