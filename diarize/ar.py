"""Speaker model from linear prediction (AR model).

A speaker is one AR model of order p, fitted by solving the normal equations R a = r.
To test a segment against a speaker, run that speaker's prediction-error filter over
it: a small error means the same voice. The log error ratio is the Itakura distance.
"""
from __future__ import annotations

import numpy as np

from .config import Config


def _toeplitz(r: np.ndarray) -> np.ndarray:
    k = np.arange(len(r))
    return r[np.abs(k[:, None] - k[None, :])]


def prediction_error_filter(r: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Solve the normal equations R a = r and return A = [1, -a1, ..., -ap]."""
    p = len(r) - 1
    R = _toeplitz(r[:p]) + reg * r[0] * np.eye(p)
    return np.concatenate([[1.0], -np.linalg.solve(R, r[1:])])


def itakura(ra: np.ndarray, rb: np.ndarray) -> float:
    """Symmetric Itakura distance between two autocorrelation sequences r[0..p]."""
    Aa, Ab = prediction_error_filter(ra), prediction_error_filter(rb)
    Ra, Rb = _toeplitz(ra), _toeplitz(rb)
    ea, eb = Aa @ Ra @ Aa, Ab @ Rb @ Ab            # each model on its own data (minimum error)
    cross_a, cross_b = Ab @ Ra @ Ab, Aa @ Rb @ Aa  # each model on the other's data
    return 0.5 * float(np.log(max(cross_a, 1e-12) / max(ea, 1e-12)) + np.log(max(cross_b, 1e-12) / max(eb, 1e-12)))


class ARStats:
    """Running sums, so models can be merged in O(p): frame count, sum of the
    normalised autocorrelation r[1..p]/r[0], and sums for the mean log-F0."""

    def __init__(self, p: int):
        self.p = p
        self.n = 0.0
        self.s = np.zeros(p)
        self.f0_n = 0.0
        self.f0_s = 0.0

    @classmethod
    def from_frames(cls, feat: np.ndarray, f0: np.ndarray) -> "ARStats":
        s = cls(feat.shape[1])
        s.add(feat, f0)
        return s

    def add(self, feat: np.ndarray, f0: np.ndarray):
        self.n += len(feat)
        self.s += feat.sum(0)
        v = f0 > 0
        self.f0_n += float(v.sum())
        self.f0_s += float(np.log(f0[v]).sum())

    def merge_(self, o: "ARStats"):
        self.n += o.n
        self.s += o.s
        self.f0_n += o.f0_n
        self.f0_s += o.f0_s

    def merged(self, o: "ARStats") -> "ARStats":
        m = self.copy()
        m.merge_(o)
        return m

    def copy(self) -> "ARStats":
        c = ARStats(self.p)
        c.merge_(self)
        return c

    def decay_to(self, max_n: float):
        """Forget old data so the model can follow slow changes."""
        if self.n > max_n:
            k = max_n / self.n
            self.n *= k
            self.s *= k
            self.f0_n *= k
            self.f0_s *= k

    @property
    def r(self) -> np.ndarray:
        """Average normalised autocorrelation r[0..p] (r[0] = 1)."""
        return np.concatenate([[1.0], self.s / max(self.n, 1.0)])

    @property
    def coeffs(self) -> np.ndarray:
        return prediction_error_filter(self.r)

    @property
    def f0_hz(self) -> float:
        return float(np.exp(self.f0_s / self.f0_n)) if self.f0_n > 0 else 0.0

    def distance(self, o: "ARStats", cfg: Config, kind: str = "cluster") -> float:
        """< 0: same speaker, > 0: different.  kind = "cluster" (segment vs model) or "merge" (model vs model)."""
        if self.n < 2 or o.n < 2:
            return -np.inf
        d = itakura(self.r, o.r)
        if self.f0_n >= 10 and o.f0_n >= 10:
            d += cfg.ar_pitch_weight * abs(self.f0_s / self.f0_n - o.f0_s / o.f0_n)
        return d - (cfg.ar_merge_thr if kind == "merge" else cfg.ar_clu_thr)


def change_score(feat_l: np.ndarray, f0_l: np.ndarray, feat_r: np.ndarray, f0_r: np.ndarray, pitch_weight: float) -> float:
    """Speaker-change score between two adjacent windows: Itakura distance + jump of the mean log-F0."""
    d = itakura(np.concatenate([[1.0], feat_l.mean(0)]), np.concatenate([[1.0], feat_r.mean(0)]))
    vl, vr = f0_l > 0, f0_r > 0
    if vl.sum() >= 10 and vr.sum() >= 10:
        d += pitch_weight * abs(np.log(f0_l[vl]).mean() - np.log(f0_r[vr]).mean())
    return d
