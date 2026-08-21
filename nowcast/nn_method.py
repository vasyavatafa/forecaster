"""Method 5: small MLP over rolling statistics of all selected series,
with RevIN (Reversible Instance Normalization, Kim et al. 2021) applied
to the raw window before it enters the network, multivariate output
(forecasts all channels jointly so cross-series correlation can be
picked up).

Refit strategy: true from-scratch refit at every one of the ~30% x N
walk-forward steps is prohibitively slow for a NN. Default behaviour is
warm-start: a full training pass the first time `.step()` is called,
then a short fine-tune (a handful of epochs on a recent+random batch of
window/target pairs, including the newest one) at every later step --
this still satisfies "refit using all history available so far" in
spirit while keeping wall-clock time bounded. Set `full_retrain=True`
for a literal from-scratch refit every step (much slower, provided for
completeness / small grids).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _rolling_features(window_block: np.ndarray) -> np.ndarray:
    """window_block: (window_len, n_channels) raw (already gap-filled)
    values. Returns a flat (n_channels * n_feat,) feature vector of
    compact rolling statistics per channel: mean, std, min, max,
    last-minus-first (trend proxy), lag-1 autocorrelation proxy."""
    feats = []
    x = window_block
    n = x.shape[0]
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    trend = (x[-1] - x[0]) / max(n - 1, 1)
    if n > 2:
        a, b = x[:-1], x[1:]
        num = ((a - a.mean(axis=0)) * (b - b.mean(axis=0))).sum(axis=0)
        den = np.sqrt(((a - a.mean(axis=0)) ** 2).sum(axis=0) * ((b - b.mean(axis=0)) ** 2).sum(axis=0)) + 1e-8
        autocorr = num / den
    else:
        autocorr = np.zeros(x.shape[1])
    feats = np.stack([mean, std, mn, mx, trend, autocorr], axis=0)  # (6, channels)
    return feats.T.reshape(-1)  # (channels*6,)


if _HAS_TORCH:

    class RevIN(nn.Module):
        def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
            super().__init__()
            self.eps = eps
            self.affine = affine
            if affine:
                self.weight = nn.Parameter(torch.ones(num_channels))
                self.bias = nn.Parameter(torch.zeros(num_channels))

        def forward(self, x: "torch.Tensor", mode: str):
            # x: (batch, window, channels)
            if mode == "norm":
                self.mean = x.mean(dim=1, keepdim=True)
                self.std = x.std(dim=1, keepdim=True) + self.eps
                out = (x - self.mean) / self.std
                if self.affine:
                    out = out * self.weight + self.bias
                return out
            elif mode == "denorm":
                # x: (batch, 1, channels) -- same shape as self.mean/self.std
                out = x
                if self.affine:
                    out = (out - self.bias) / (self.weight + self.eps)
                out = out * self.std + self.mean
                return out
            raise ValueError(mode)

    class _Net(nn.Module):
        def __init__(self, window: int, n_channels: int, n_feat_per_ch: int, hidden: int):
            super().__init__()
            self.window = window
            self.n_channels = n_channels
            self.revin = RevIN(n_channels)
            in_dim = window * n_channels + n_feat_per_ch * n_channels
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, n_channels),
            )

        def forward(self, raw_window: "torch.Tensor", feats: "torch.Tensor"):
            # raw_window: (batch, window, channels), feats: (batch, channels*n_feat)
            norm_win = self.revin(raw_window, "norm")
            flat = norm_win.reshape(norm_win.shape[0], -1)
            x = torch.cat([flat, feats], dim=1)
            out_norm = self.net(x)  # (batch, channels), in normalized space
            out = self.revin(out_norm.unsqueeze(1), "denorm").squeeze(1)
            return out


@dataclass
class NNRevINMethod:
    window: int = 26
    hidden: int = 64
    epochs_init: int = 60
    epochs_refit: int = 5
    lr: float = 1e-3
    batch_history: int = 64  # max number of (window,target) pairs sampled per refit
    full_retrain: bool = False
    seed: int = 0

    param_grid = {
        "window": [12, 26, 52],
        "hidden": [32, 64, 128],
    }
    display_name = "nn_revin"

    def __post_init__(self):
        if not _HAS_TORCH:
            raise ImportError("torch is required for NNRevINMethod (pip install torch)")
        self._net = None
        self._columns = None
        self._n_feat = 6
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _ensure_net(self, n_channels: int):
        if self._net is None:
            self._net = _Net(self.window, n_channels, self._n_feat, self.hidden)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

    def _build_examples(self, filled: np.ndarray, max_examples: int | None) -> tuple[np.ndarray, np.ndarray]:
        """filled: (T, channels) fully gap-filled array. Builds all
        (window -> next value) training pairs available in it."""
        w = self.window
        T = filled.shape[0]
        if T <= w:
            return np.empty((0, w, filled.shape[1])), np.empty((0, filled.shape[1]))
        starts = np.arange(0, T - w)
        if max_examples is not None and len(starts) > max_examples:
            recent_k = max_examples // 2
            recent = starts[-recent_k:]
            rest = starts[:-recent_k]
            extra = np.random.choice(rest, size=max_examples - recent_k, replace=False) if len(rest) else np.array([], dtype=int)
            starts = np.sort(np.concatenate([extra, recent]))
        X = np.stack([filled[s : s + w] for s in starts])
        y = np.stack([filled[s + w] for s in starts])
        return X, y

    def _train(self, X: np.ndarray, y: np.ndarray, epochs: int):
        if len(X) == 0:
            return
        Xt = torch.tensor(X, dtype=torch.float32)
        Ft = torch.tensor(np.stack([_rolling_features(x) for x in X]), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self._net.train()
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            self._opt.zero_grad()
            pred = self._net(Xt, Ft)
            loss = loss_fn(pred, yt)
            loss.backward()
            self._opt.step()

    def step(self, history: pd.DataFrame) -> pd.Series:
        cols = list(history.columns)
        out = pd.Series(np.nan, index=cols)
        if len(history) <= self.window + 5:
            return out

        filled = history.interpolate(limit=5, limit_direction="both").ffill().bfill()
        if filled.isna().any().any():
            return out  # a column never observed yet -- skip this step
        arr = filled.values.astype(np.float32)

        self._ensure_net(len(cols))
        if self._columns is not None and self._columns != cols:
            self._net = None
            self._ensure_net(len(cols))
        self._columns = cols

        if self.full_retrain:
            self._net = None
            self._ensure_net(len(cols))
            X, y = self._build_examples(arr[:-1], max_examples=None)
            self._train(X, y, self.epochs_init)
        elif self._net is not None and getattr(self, "_fitted_once", False):
            X, y = self._build_examples(arr[:-1], max_examples=self.batch_history)
            self._train(X, y, self.epochs_refit)
        else:
            X, y = self._build_examples(arr[:-1], max_examples=self.batch_history * 4)
            self._train(X, y, self.epochs_init)
            self._fitted_once = True

        self._net.eval()
        last_window = arr[-self.window :]
        with torch.no_grad():
            Xt = torch.tensor(last_window[None, :, :], dtype=torch.float32)
            Ft = torch.tensor(_rolling_features(last_window)[None, :], dtype=torch.float32)
            pred = self._net(Xt, Ft).numpy()[0]
        out.loc[cols] = pred
        return out
