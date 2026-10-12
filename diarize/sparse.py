"""Speaker model from sparse coding with a learned dictionary.

One person makes many different sounds, so each speaker gets a small dictionary of
spectral-envelope shapes ("atoms"), learned from their own frames with K-SVD.
OMP then describes a frame with the best atom. What is left over (the residual)
is small for the right speaker and large for anyone else.
"""
from __future__ import annotations

import numpy as np

from .ar import ARStats
from .config import Config


def envelope(feat: np.ndarray, nfreq: int = 32, reg: float = 1e-4) -> np.ndarray:
    """Log spectral envelope of each frame's AR model at `nfreq` frequencies, mean removed."""
    n, p = feat.shape
    k = np.arange(p)
    r = np.concatenate([np.ones((n, 1)), feat], axis=1)
    R = r[:, np.abs(k[:, None] - k[None, :])] + reg * np.eye(p)
    a = np.linalg.solve(R, feat[:, :, None])[:, :, 0]
    w = np.pi * (np.arange(nfreq) + 0.5) / nfreq
    E = np.exp(-1j * np.outer(w, np.arange(1, p + 1)))
    L = -np.log(np.abs(1.0 - a @ E.T) ** 2 + 1e-9)
    return L - L.mean(axis=1, keepdims=True)


def omp(D: np.ndarray, X: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal matching pursuit for all rows of X at once.

    Repeat T times: pick the atom that matches the residual best, then re-fit the
    coefficients of the picked atoms. Returns (coefficients, residual).
    
    """
    n, K = len(X), len(D)
    G = D @ D.T
    R = X.copy()
    sel = np.zeros((n, 0), dtype=int)
    coef = np.zeros((n, 0))
    for t in range(min(T, K)):
        c = np.abs(R @ D.T)
        if t:
            np.put_along_axis(c, sel, -1.0, axis=1)          # never pick the same atom twice
        sel = np.concatenate([sel, c.argmax(axis=1)[:, None]], axis=1)
        Ds = D[sel]
        Gs = G[sel[:, :, None], sel[:, None, :]] + 1e-8 * np.eye(t + 1)
        coef = np.linalg.solve(Gs, np.einsum("ntd,nd->nt", Ds, X)[:, :, None])[:, :, 0]
        R = X - np.einsum("nt,ntd->nd", coef, Ds)
    A = np.zeros((n, K))
    np.put_along_axis(A, sel, coef, axis=1)
    return A, R


def ksvd(X: np.ndarray, K: int, T: int, iters: int = 6, seed: int = 0) -> np.ndarray:
    """Learn K atoms from the rows of X.

    Repeat: (1) code every frame with OMP, (2) update each atom as the best
    rank-1 fit (SVD) of the error of the frames that use it.
    
    """
    rng = np.random.default_rng(seed)
    D = X[rng.choice(len(X), K, replace=len(X) < K)].copy()
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    for _ in range(iters):
        A, R = omp(D, X, T)
        for k in range(K):
            use = np.flatnonzero(A[:, k])
            if len(use) < 2:                                  # unused atom: restart it on the worst-coded frame
                D[k] = X[np.argmax(np.sum(R ** 2, axis=1))]
                D[k] /= np.linalg.norm(D[k]) + 1e-9
                continue
            E = R[use] + np.outer(A[use, k], D[k])
            U, s, Vt = np.linalg.svd(E, full_matrices=False)
            D[k] = Vt[0]
            A[use, k] = U[:, 0] * s[0]
            R[use] = E - np.outer(A[use, k], D[k])
    return D


def residual_ratio(D: np.ndarray, X: np.ndarray, T: int) -> float:
    """Share of the energy of the frames X that dictionary D cannot represent (0 = perfect)."""
    _, R = omp(D, X, T)
    return float(np.sum(R ** 2) / (np.sum(X ** 2) + 1e-12))


class DictStats(ARStats):
    """ARStats (autocorrelation and F0 sums) plus the speaker's recent envelope frames and
    the dictionary learned from them (re-learned lazily after the frames change)."""

    MAX_FRAMES = 2000       # most recent frames kept per speaker for learning the dictionary
    ORDER, ATOMS, SPARSITY = 16, 16, 1   # set from the Config by gauss.stats_class

    def __init__(self, p: int, nfreq: int = 32):
        super().__init__(p)
        self.X = np.zeros((0, nfreq))
        self._D = None

    @classmethod
    def from_frames(cls, feat: np.ndarray, f0: np.ndarray) -> "DictStats":
        p = cls.ORDER
        s = cls(p, feat.shape[1] - p)
        ARStats.add(s, feat[:, :p], f0)
        s.X = feat[:, p:]
        return s

    def merge_(self, o: "DictStats"):
        super().merge_(o)
        self.X = np.concatenate([self.X, o.X])[-self.MAX_FRAMES:]
        self._D = None

    def copy(self) -> "DictStats":
        c = DictStats(self.p, self.X.shape[1])
        c.merge_(self)
        return c

    def merged(self, o: "DictStats") -> "DictStats":
        m = self.copy()
        m.merge_(o)
        return m

    @property
    def dictionary(self) -> np.ndarray:
        if self._D is None:
            self._D = ksvd(self.X, min(self.ATOMS, max(len(self.X) // 4, 1)), self.SPARSITY)
        return self._D

    def distance(self, o: "DictStats", cfg: Config, kind: str = "cluster") -> float:
        """< 0: same speaker, > 0: different.

        "cluster": code this segment's frames with speaker o's dictionary.
        "merge": two speaker models, each coded with the other's dictionary.
        
        """
        if len(self.X) < 2 or len(o.X) < 2:
            return -np.inf
        if kind == "merge":
            a, b = self.X[-500:], o.X[-500:]
            d = 0.5 * (residual_ratio(o.dictionary, a, self.SPARSITY) + residual_ratio(self.dictionary, b, self.SPARSITY))
        else:
            d = residual_ratio(o.dictionary, self.X, self.SPARSITY)
        if self.f0_n >= 10 and o.f0_n >= 10:
            d += cfg.dict_pitch_weight * abs(self.f0_s / self.f0_n - o.f0_s / o.f0_n)
        return d - (cfg.dict_merge_thr if kind == "merge" else cfg.dict_thr)


def change_score(X_l: np.ndarray, f0_l: np.ndarray, X_r: np.ndarray, f0_r: np.ndarray,
                 atoms: int, pitch_weight: float) -> float:
    """Speaker-change score between two adjacent windows.

    Learn a small dictionary on each window and code the other window with it.
    Same speaker: about as good as on its own window. New speaker: clearly worse.
    The jump of the mean log-F0 is added.
    
    """
    D_l, D_r = ksvd(X_l, atoms, 1, iters=4), ksvd(X_r, atoms, 1, iters=4)
    d = 0.5 * (residual_ratio(D_l, X_r, 1) + residual_ratio(D_r, X_l, 1)
               - residual_ratio(D_l, X_l, 1) - residual_ratio(D_r, X_r, 1))
    vl, vr = f0_l > 0, f0_r > 0
    if vl.sum() >= 10 and vr.sum() >= 10:
        d += pitch_weight * abs(np.log(f0_l[vl]).mean() - np.log(f0_r[vr]).mean())
    return d
