"""Clean a segment before it is saved.

1. remove noise, using the last silence as the noise sample (Wiener filter, or spectral subtraction)
2. trim the quiet ends
3. set a fixed loudness and add short fades
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import Config

_NPERSEG, _HOP = 512, 128


def noise_psd(noise: np.ndarray, fs: int) -> np.ndarray | None:
    if len(noise) < _NPERSEG * 2:
        return None
    _, _, Z = signal.stft(noise, fs, window="hann", nperseg=_NPERSEG, noverlap=_NPERSEG - _HOP)
    return np.mean(np.abs(Z) ** 2, axis=1)


def denoise(x: np.ndarray, fs: int, npsd: np.ndarray | None, method="wiener", oversub=2.0, floor=0.05) -> np.ndarray:
    if npsd is None or len(x) < _NPERSEG:
        return x
    _, _, Z = signal.stft(x, fs, window="hann", nperseg=_NPERSEG, noverlap=_NPERSEG - _HOP)
    P = np.abs(Z) ** 2
    if method == "wiener":
        # Wiener gain per frequency: H = S / (S + N), with S estimated as P - N
        gain = np.maximum(1.0 - npsd[:, None] / np.maximum(P, 1e-20), floor)
    else:
        gain = np.sqrt(np.maximum(P - oversub * npsd[:, None], floor * P) / np.maximum(P, 1e-20))
    # smooth the gain over time so it does not flutter
    gain = signal.lfilter([0.5], [1, -0.5], gain, axis=1)
    _, y = signal.istft(Z * gain, fs, window="hann", nperseg=_NPERSEG, noverlap=_NPERSEG - _HOP)
    return y[: len(x)]


def trim(x: np.ndarray, fs: int, rel_db: float = -35.0, keep_s: float = 0.05) -> np.ndarray:
    hop = int(0.01 * fs)
    if len(x) < 4 * hop:
        return x
    e = np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x) - hop, hop)])
    e_db = 10 * np.log10(e + 1e-12)
    on = np.where(e_db > e_db.max() + rel_db)[0]
    if len(on) == 0:
        return x
    k = int(keep_s * fs)
    return x[max(on[0] * hop - k, 0): min((on[-1] + 1) * hop + k, len(x))]


def normalise(x: np.ndarray, fs: int, target_dbfs: float = -20.0, fade_s: float = 0.01) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0.99:
        y *= 0.99 / peak
    n = min(int(fade_s * fs), len(y) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def clean_segment(x: np.ndarray, fs: int, noise: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = denoise(x, fs, noise_psd(noise, fs), cfg.clean_method, cfg.clean_oversub, cfg.clean_floor)
    y = trim(y, fs)
    return normalise(y, fs, cfg.clean_target_dbfs).astype(np.float32)
