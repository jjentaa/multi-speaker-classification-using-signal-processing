"""Gaussian sufficient statistics and the Bayesian Information Criterion.

Both the speaker-change detector and the speaker clustering ask the same
question: "are these two blocks of frames better modelled by ONE Gaussian or
by TWO?".  With BIC:

    dBIC = N/2 log|S| - N1/2 log|S1| - N2/2 log|S2| - lambda * P
    P    = 1/2 (d + d(d+1)/2) log N          (extra parameters of the 2-model)

dBIC > 0  -> two different speakers,   dBIC < 0 -> same speaker.
"""
from __future__ import annotations

import numpy as np


class GaussStats:
    """Running (N, sum x, sum x x^T) so models can be merged in O(d^2)."""

    def __init__(self, d: int):
        self.d = d
        self.n = 0.0
        self.s1 = np.zeros(d)
        self.s2 = np.zeros((d, d))

    @classmethod
    def from_data(cls, X: np.ndarray) -> "GaussStats":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        g = cls(X.shape[1])
        g.add(X)
        return g

    def add(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if len(X) == 0:
            return
        self.n += len(X)
        self.s1 += X.sum(0)
        self.s2 += X.T @ X

    def merged(self, other: "GaussStats") -> "GaussStats":
        g = GaussStats(self.d)
        g.n, g.s1, g.s2 = self.n + other.n, self.s1 + other.s1, self.s2 + other.s2
        return g

    def merge_(self, other: "GaussStats"):
        self.n += other.n
        self.s1 += other.s1
        self.s2 += other.s2

    def decay_to(self, max_n: float):
        """Forget old data so the model can follow slow changes."""
        if self.n > max_n:
            k = max_n / self.n
            self.n *= k
            self.s1 *= k
            self.s2 *= k

    @property
    def mean(self) -> np.ndarray:
        return self.s1 / max(self.n, 1.0)

    def cov(self, reg: float = 1e-3) -> np.ndarray:
        m = self.mean
        C = self.s2 / max(self.n, 1.0) - np.outer(m, m)
        return C + reg * np.eye(self.d)

    def logdet(self) -> float:
        sign, ld = np.linalg.slogdet(self.cov())
        return ld if sign > 0 else 1e6


def delta_bic(a: GaussStats, b: GaussStats, lam: float = 1.0) -> float:
    if a.n < 2 or b.n < 2:
        return -np.inf
    ab = a.merged(b)
    d = a.d
    P = 0.5 * (d + d * (d + 1) / 2) * np.log(ab.n)
    return 0.5 * (ab.n * ab.logdet() - a.n * a.logdet() - b.n * b.logdet()) - lam * P


class SpeakerStats:
    """MFCC model (full-covariance Gaussian) + log-F0 model (1-D Gaussian)."""

    def __init__(self, d: int):
        self.mfcc = GaussStats(d)
        self.logf0 = GaussStats(1)

    @classmethod
    def from_frames(cls, mfcc: np.ndarray, f0: np.ndarray) -> "SpeakerStats":
        s = cls(mfcc.shape[1])
        s.add(mfcc, f0)
        return s

    def add(self, mfcc: np.ndarray, f0: np.ndarray):
        self.mfcc.add(mfcc)
        v = f0 > 0
        self.logf0.add(np.log(f0[v]))

    def merge_(self, o: "SpeakerStats"):
        self.mfcc.merge_(o.mfcc)
        self.logf0.merge_(o.logf0)

    def merged(self, o: "SpeakerStats") -> "SpeakerStats":
        s = SpeakerStats(self.mfcc.d)
        s.mfcc, s.logf0 = self.mfcc.merged(o.mfcc), self.logf0.merged(o.logf0)
        return s

    def decay_to(self, n: float):
        k = n / max(self.mfcc.n, 1.0)
        if k < 1:
            self.mfcc.decay_to(self.mfcc.n * k)
            self.logf0.decay_to(self.logf0.n * k)

    @property
    def n(self) -> float:
        return self.mfcc.n

    @property
    def f0_hz(self) -> float:
        return float(np.exp(self.logf0.mean[0])) if self.logf0.n > 0 else 0.0


def speaker_distance(a: SpeakerStats, b: SpeakerStats, lam: float = 1.0, pitch_weight: float = 1.0) -> float:
    """dBIC on MFCCs plus dBIC on log-F0 (features assumed independent)."""
    d = delta_bic(a.mfcc, b.mfcc, lam)
    if a.logf0.n >= 10 and b.logf0.n >= 10:
        d += pitch_weight * delta_bic(a.logf0, b.logf0, lam)
    return d
