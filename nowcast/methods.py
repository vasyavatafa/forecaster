"""The 5 one-step-ahead forecasting methods used to fill in the lagged
signal x_t with an estimate hat_x_t.

Every method exposes a uniform `.step(history) -> forecast` interface so
the walk-forward backtest engine (backtest.py) can drive all of them the
same way:

  * univariate methods (`RollingMeanMethod`, `LinearTrendMethod`,
    `BayesianARMethod`) take `history: pd.Series` (one column, values up
    to and including t, may contain NaNs) and return a single float
    forecast for t+1.
  * multivariate methods (`VARJointMethod`, and the NN in nn_method.py)
    take `history: pd.DataFrame` (all selected columns up to and
    including t) and return a `pd.Series` of forecasts for t+1, indexed
    by column, so cross-series correlation can be used.

All methods handle short interior gaps by linear interpolation within
the trailing window (`interp_limit` weeks) rather than dropping rows,
so lag structure / time spacing stays correct. A method returns NaN (or
an all-NaN row) when there isn't enough history yet -- the backtest
engine simply skips those points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _prepare_window(
    history: pd.Series, window: int | None, min_periods: int, interp_limit: int = 3
) -> pd.Series | None:
    """Trailing window (or full history if window is None), short gaps
    linearly interpolated. Returns None if fewer than `min_periods`
    genuinely observed points are available."""
    block = history if window is None else history.iloc[-window:]
    n_real = int(block.notna().sum())
    if n_real < min_periods:
        return None
    filled = block.interpolate(limit=interp_limit, limit_direction="both")
    if filled.isna().any():
        filled = filled.ffill().bfill()
    return filled


def _week_offsets(index: pd.DatetimeIndex) -> np.ndarray:
    t0 = index[0]
    return np.array([(d - t0).days / 7.0 for d in index])


# ---------------------------------------------------------------------------
# 1. simple moving average in a trailing window
# ---------------------------------------------------------------------------

@dataclass
class RollingMeanMethod:
    window: int = 8
    min_periods: int | None = None

    param_grid = {"window": [4, 8, 12, 16, 26, 52]}
    display_name = "rolling_mean"

    def __post_init__(self):
        if self.min_periods is None:
            self.min_periods = max(2, self.window // 2)

    def step(self, history: pd.Series) -> float:
        w = _prepare_window(history, self.window, self.min_periods)
        if w is None:
            return np.nan
        return float(w.mean())


# ---------------------------------------------------------------------------
# 2. linear trend fit in a trailing window, extrapolated 1 step ahead
# ---------------------------------------------------------------------------

@dataclass
class LinearTrendMethod:
    window: int = 12
    min_periods: int | None = None
    robust: bool = False  # Theil-Sen instead of OLS

    param_grid = {"window": [4, 8, 12, 16, 26, 52], "robust": [False, True]}
    display_name = "linear_trend"

    def __post_init__(self):
        if self.min_periods is None:
            self.min_periods = max(3, self.window // 2)

    def step(self, history: pd.Series) -> float:
        w = _prepare_window(history, self.window, max(3, self.min_periods))
        if w is None:
            return np.nan
        t = _week_offsets(w.index)
        y = w.values.astype(float)
        if self.robust:
            from scipy.stats import theilslopes

            slope, intercept, _, _ = theilslopes(y, t)
        else:
            slope, intercept = np.polyfit(t, y, 1)
        t_next = t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1.0)
        return float(intercept + slope * t_next)


# ---------------------------------------------------------------------------
# 3. VAR / VARMAX multivariate statistical model ("VARIMA": VAR + optional
#    differencing (I) + optional MA term)
# ---------------------------------------------------------------------------

@dataclass
class VARJointMethod:
    window: int | None = 52
    p: int = 2
    d: int = 0
    q: int = 0
    trend: str = "c"
    min_obs_per_col: int | None = None

    param_grid = {
        "window": [26, 52, 104, None],
        "p": [1, 2, 4],
        "d": [0, 1],
        "trend": ["c", "n"],
    }
    display_name = "var_multivariate"

    def __post_init__(self):
        if self.min_obs_per_col is None:
            self.min_obs_per_col = max(3 * self.p + self.d + 5, 12)

    def step(self, history: pd.DataFrame) -> pd.Series:
        cols = list(history.columns)
        out = pd.Series(np.nan, index=cols)

        block = history if self.window is None else history.iloc[-self.window :]
        valid_cols = [c for c in cols if block[c].notna().sum() >= self.min_obs_per_col]
        if len(valid_cols) < 1:
            return out

        sub = block[valid_cols].interpolate(limit=3, limit_direction="both").dropna()
        if len(sub) < self.min_obs_per_col:
            return out

        levels_last = sub.iloc[-1].copy()
        data = sub
        if self.d == 1:
            data = data.diff().dropna()
        if len(data) < self.p + 5:
            return out

        try:
            if self.q == 0:
                from statsmodels.tsa.api import VAR

                model = VAR(data)
                res = model.fit(maxlags=self.p, trend=self.trend, ic=None)
                y_hist = data.values[-res.k_ar :] if res.k_ar > 0 else data.values[-1:]
                fc = res.forecast(y_hist, steps=1)[0]
            else:
                from statsmodels.tsa.statespace.varmax import VARMAX

                model = VARMAX(data, order=(self.p, self.q), trend=self.trend)
                res = model.fit(disp=False, maxiter=50)
                fc = res.forecast(steps=1).values[0]
        except Exception:
            return out

        fc = pd.Series(fc, index=valid_cols)
        if self.d == 1:
            fc = fc + levels_last
        out.loc[valid_cols] = fc
        return out


# ---------------------------------------------------------------------------
# 4. Bayesian AR: BayesianRidge (evidence-maximization / Normal-Gamma
#    conjugate prior on the weights) on lagged values + optional local
#    linear trend term.
#
# Why Bayesian here specifically: windows are short and noisy (weekly
# macro data with gaps and a handful of shock episodes), and a plain OLS
# AR(p) overfits those short windows. BayesianRidge puts a Gaussian prior
# on the AR coefficients and infers its precision from the data (ARD-like
# regularization strength), which shrinks coefficients toward 0 exactly
# when the window doesn't have enough signal to support them, and widens
# again once more history is available -- without hand-tuning a ridge
# penalty per series/window. It also returns a predictive variance, which
# is useful downstream for confidence-weighting hat_x_t in the event rule.
# ---------------------------------------------------------------------------

@dataclass
class BayesianARMethod:
    window: int | None = 52
    lags: int = 4
    trend: bool = True
    min_periods: int | None = None

    param_grid = {
        "window": [16, 26, 52, 104, None],
        "lags": [1, 2, 4, 8],
        "trend": [False, True],
    }
    display_name = "bayesian_ar"

    def __post_init__(self):
        if self.min_periods is None:
            self.min_periods = max(self.lags * 3 + 5, 10)

    def step(self, history: pd.Series) -> float:
        w = _prepare_window(history, self.window, self.min_periods)
        if w is None or len(w) <= self.lags + 3:
            return np.nan

        y = w.values.astype(float)
        t = _week_offsets(w.index)
        n = len(y)
        p = self.lags

        rows = []
        targets = []
        for i in range(p, n):
            feat = list(y[i - p : i][::-1])  # most recent lag first
            if self.trend:
                feat.append(t[i])
            rows.append(feat)
            targets.append(y[i])
        X = np.array(rows)
        yy = np.array(targets)
        if len(yy) < 5:
            return np.nan

        model = BayesianRidge(compute_score=False)
        model.fit(X, yy)

        feat_next = list(y[n - p : n][::-1])
        if self.trend:
            t_next = t[-1] + (t[-1] - t[-2] if n > 1 else 1.0)
            feat_next.append(t_next)
        pred = model.predict(np.array(feat_next).reshape(1, -1))[0]
        return float(pred)

    def step_with_std(self, history: pd.Series) -> tuple[float, float]:
        """Same as .step but also returns the posterior predictive std,
        for callers that want an uncertainty-aware hat_x_t."""
        w = _prepare_window(history, self.window, self.min_periods)
        if w is None or len(w) <= self.lags + 3:
            return np.nan, np.nan
        y = w.values.astype(float)
        t = _week_offsets(w.index)
        n = len(y)
        p = self.lags
        rows, targets = [], []
        for i in range(p, n):
            feat = list(y[i - p : i][::-1])
            if self.trend:
                feat.append(t[i])
            rows.append(feat)
            targets.append(y[i])
        X, yy = np.array(rows), np.array(targets)
        if len(yy) < 5:
            return np.nan, np.nan
        model = BayesianRidge(compute_score=False)
        model.fit(X, yy)
        feat_next = list(y[n - p : n][::-1])
        if self.trend:
            t_next = t[-1] + (t[-1] - t[-2] if n > 1 else 1.0)
            feat_next.append(t_next)
        pred, std = model.predict(np.array(feat_next).reshape(1, -1), return_std=True)
        return float(pred[0]), float(std[0])


METHOD_REGISTRY = {
    "rolling_mean": RollingMeanMethod,
    "linear_trend": LinearTrendMethod,
    "var_multivariate": VARJointMethod,
    "bayesian_ar": BayesianARMethod,
}

try:
    from .nn_method import NNRevINMethod

    METHOD_REGISTRY["nn_revin"] = NNRevINMethod
except ImportError:  # torch not installed
    pass

MULTIVARIATE_METHODS = {"var_multivariate", "nn_revin"}
