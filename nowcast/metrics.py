"""Forecast-quality metrics for one-step-ahead nowcasts.

All functions take arrays of equal length with actual (y_true) and
predicted (y_pred) values. NaNs are dropped pairwise before computing
anything, since walk-forward loops can produce occasional missing
predictions (e.g. a step skipped for lack of history).
"""
from __future__ import annotations

import numpy as np


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, in percent, bounded in [0, 200]."""
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    denom = (np.abs(yt) + np.abs(yp))
    out = np.zeros_like(yt)
    nonzero = denom > 1e-12
    out[nonzero] = np.abs(yt[nonzero] - yp[nonzero]) / denom[nonzero]
    return float(100.0 * np.mean(out) * 2)


def r2(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    if yt.size < 2:
        return np.nan
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def mase(y_true, y_pred, y_history, season: int = 1) -> float:
    """Mean Absolute Scaled Error vs. a naive seasonal-lag benchmark.

    y_history is the full in-sample history used to compute the scale
    (mean absolute seasonal difference). MASE < 1 means the model beats
    the naive "repeat value from `season` steps ago" forecast.
    """
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    hist = np.asarray(y_history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if hist.size <= season:
        return np.nan
    naive_diffs = np.abs(hist[season:] - hist[:-season])
    scale = np.mean(naive_diffs)
    if scale < 1e-12:
        return np.nan
    return float(np.mean(np.abs(yt - yp)) / scale)


def directional_accuracy(y_true, y_pred, y_prev) -> float:
    """Share of steps where the predicted direction of change (up/down/flat
    relative to the previous actual value) matches the realized direction.
    Relevant for event identification, where the sign of the move often
    matters more than its exact magnitude.
    """
    yt, yp = _clean(y_true, y_pred)
    yprev = np.asarray(y_prev, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(yprev)
    yt, yp, yprev = np.asarray(y_true)[mask], np.asarray(y_pred)[mask], yprev[mask]
    if yt.size == 0:
        return np.nan
    true_dir = np.sign(yt - yprev)
    pred_dir = np.sign(yp - yprev)
    return float(np.mean(true_dir == pred_dir))


METRIC_FUNCS = {
    "rmse": rmse,
    "mae": mae,
    "smape": smape,
    "r2": r2,
}


def compute_basic_metrics(y_true, y_pred) -> dict:
    return {name: fn(y_true, y_pred) for name, fn in METRIC_FUNCS.items()}
