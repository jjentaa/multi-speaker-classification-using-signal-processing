"""Signal-processing blocks.

Every filter keeps its state, so audio can be pushed in small chunks and the
result is the same as filtering the whole signal at once.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import dct


# ---------------------------------------------------------------------------
# 2.1.2 Low-pass (FIR anti-aliasing) + 2.1.3 Down-sampling
# ---------------------------------------------------------------------------
class Decimator:
    """Low-pass FIR, then keep every M-th sample. The filter prevents aliasing."""

    def __init__(self, fs_in: int, fs_out: int, numtaps: int = 97, cutoff_ratio: float = 0.9,
                 h: np.ndarray | None = None):
        if fs_in % fs_out:
            raise ValueError(f"fs_in ({fs_in}) must be an integer multiple of fs_out ({fs_out})")
        self.M = fs_in // fs_out
        if self.M == 1:
            self.h = np.array([1.0])
        elif h is not None:
            self.h = h
        else:
            self.h = signal.firwin(numtaps, cutoff_ratio * (fs_out / 2), fs=fs_in, window="hamming")
        self.zi = np.zeros(len(self.h) - 1)
        self.phase = 0  # position of the next sample to keep, inside the next chunk

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.M == 1:
            return x.astype(np.float64)
        y, self.zi = signal.lfilter(self.h, 1.0, x, zi=self.zi)
        out = y[self.phase::self.M]
        self.phase = (self.phase - len(x)) % self.M
        return out


def kaiser_lowpass(passband_hz: float, stopband_hz: float, ripple: float, fs: float) -> np.ndarray:
    """Design a low-pass FIR with the Kaiser window method.

    A = -20 log10(ripple)             attenuation needed (dB)
    beta = 0.1102 (A - 8.7)           window shape (for A > 50)
    M = (A - 8) / (2.285 dw)          filter order, dw = stopband edge - passband edge
    h[n] = ideal low-pass (sinc) x Kaiser window
    
    """
    wp, ws = 2 * np.pi * passband_hz / fs, 2 * np.pi * stopband_hz / fs
    A = -20 * np.log10(ripple)
    beta = 0.1102 * (A - 8.7) if A > 50 else 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21) if A >= 21 else 0.0
    M = int(np.ceil((A - 8) / (2.285 * (ws - wp))))
    M += M % 2                                   # even order, so the filter is symmetric (linear phase)
    m = np.arange(M + 1) - M / 2
    wc = (wp + ws) / 2
    hd = (wc / np.pi) * np.sinc(wc * m / np.pi)
    w = np.i0(beta * np.sqrt(1 - (m / (M / 2)) ** 2)) / np.i0(beta)
    return hd * w


# ---------------------------------------------------------------------------
# 2.1.4 Pre-emphasis   y[n] = x[n] - a x[n-1]   (first-order FIR high-pass)
# ---------------------------------------------------------------------------
class PreEmphasis:
    def __init__(self, a: float = 0.97):
        self.a = a
        self.prev = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        y = np.empty_like(x)
        y[0] = x[0] - self.a * self.prev
        y[1:] = x[1:] - self.a * x[:-1]
        self.prev = x[-1]
        return y


# ---------------------------------------------------------------------------
# 2.1.5 IIR Butterworth (second-order sections for numerical stability)
# ---------------------------------------------------------------------------
class Butterworth:
    def __init__(self, order: int, cutoff, fs: int, btype: str):
        self.sos = signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


# ---------------------------------------------------------------------------
# Median filter (non-linear): removes isolated outliers without smearing edges
# ---------------------------------------------------------------------------
class MedianFilter:
    """Causal running median over the last N values (N odd), stateful."""

    def __init__(self, N: int):
        self.N = N
        self.hist = np.zeros(N - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.asarray(x, dtype=float)
        buf = np.concatenate([self.hist, np.asarray(x, dtype=float)])
        self.hist = buf[len(buf) - (self.N - 1):]
        return np.median(np.lib.stride_tricks.sliding_window_view(buf, self.N), axis=1)


# ---------------------------------------------------------------------------
# Adaptive filtering (NLMS)   e[n] = d[n] - h^T x[n]
# ---------------------------------------------------------------------------
class AdaptiveNoiseCanceller:
    """NLMS filter that removes hum using a single microphone.

    The reference is the microphone itself, delayed by D samples. A hum repeats, so
    it can be predicted from the past and subtracted:  e[n] = d[n] - h^T x[n].
    The weights are only updated while nobody is talking; otherwise the filter
    would learn to cancel the speech too.
    
    """

    def __init__(self, taps: int = 512, delay: int = 480, mu: float = 0.02, fs: int = 16000,
                 gate_db: float = 6.0, eps: float = 1e-6):
        self.L, self.mu, self.eps = taps, mu, eps
        self.h = np.zeros(taps)                     # filter weights
        self.hist = np.zeros(taps - 1 + delay)      # past microphone samples
        self.a = np.exp(-1.0 / (0.010 * fs))        # smoothing of the output power (10 ms)
        self.gate = 10 ** (gate_db / 10)
        self.rise = 10 ** (3.0 / 10 / fs)           # the noise floor may rise 3 dB per second
        self.p = 0.0                                # smoothed output power
        self.floor = 0.0                            # noise floor of the output
        self.warmup = int(0.05 * fs)                # skip the first 50 ms
        self.adapting = 0                           # how many samples the weights were updated on

    def process(self, d: np.ndarray) -> np.ndarray:
        L, mu, eps, h, a = self.L, self.mu, self.eps, self.h, self.a
        buf = np.concatenate([self.hist, d])
        e = np.empty(len(d))
        p, floor, warmup = self.p, self.floor, self.warmup
        for n in range(len(d)):
            x = buf[n:n + L]
            e[n] = d[n] - h @ x
            p = a * p + (1 - a) * e[n] * e[n]
            if warmup:
                warmup -= 1
                floor = p
                continue
            floor = p if p < floor else floor * self.rise
            if p < floor * self.gate:
                h += (mu * e[n] / (eps + x @ x + L * e[n] * e[n])) * x
                self.adapting += 1
        self.p, self.floor, self.warmup = p, floor, warmup
        self.hist = buf[len(d):]
        return e


# ---------------------------------------------------------------------------
# 2.1.6 Windowing
# ---------------------------------------------------------------------------
def hamming(N: int) -> np.ndarray:
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def frame_signal(x: np.ndarray, win_len: int, hop: int) -> np.ndarray:
    """Split x into overlapping frames, shape (n_frames, win_len)."""
    if len(x) < win_len:
        return np.zeros((0, win_len))
    n = 1 + (len(x) - win_len) // hop
    idx = np.arange(win_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# ---------------------------------------------------------------------------
# 2.1.7 DFT
# ---------------------------------------------------------------------------
def dft(x: np.ndarray) -> np.ndarray:
    """DFT straight from the definition, O(N^2). Only used to check that the FFT gives the same result."""
    N = len(x)
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)
    return W @ x


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """2.1.8  P[k] = |X[k]|^2 / N for k = 0..N/2 (one-sided periodogram)."""
    X = np.fft.rfft(frames, n=n_fft, axis=-1)
    return (np.abs(X) ** 2) / n_fft


def spectrogram(x: np.ndarray, fs: int, win_len=512, hop=160, n_fft=512):
    """Power spectrogram in dB. Returns (times, freqs, S_db[freq, time])."""
    frames = frame_signal(x, win_len, hop) * hamming(win_len)
    P = power_spectrum(frames, n_fft)
    t = (np.arange(len(frames)) * hop + win_len / 2) / fs
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    return t, f, 10 * np.log10(P.T + 1e-12)


# ---------------------------------------------------------------------------
# Mel filterbank + MFCC (speaker features computed from the power spectrum)
# ---------------------------------------------------------------------------
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, fs: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale, shape (n_mels, n_fft//2+1)."""
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, c, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def mfcc_from_power(P: np.ndarray, fb: np.ndarray, n_mfcc: int) -> np.ndarray:
    """log mel energies -> DCT-II -> first n_mfcc cepstral coefficients."""
    mel_e = np.log(P @ fb.T + 1e-10)
    return dct(mel_e, type=2, axis=-1, norm="ortho")[..., :n_mfcc]


# ---------------------------------------------------------------------------
# 2.1.9 Autocorrelation  (Wiener-Khinchin: r = IDFT{|X|^2})
# ---------------------------------------------------------------------------
def autocorrelation(frames: np.ndarray) -> np.ndarray:
    """Autocorrelation of each frame, computed with the FFT (zero-padded so it does not wrap around)."""
    N = frames.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.rfft(frames, n=nfft, axis=-1)
    r = np.fft.irfft(np.abs(X) ** 2, n=nfft, axis=-1)
    return r[..., :N]


def autocorr_lags(frames: np.ndarray, p: int) -> np.ndarray:
    """r[k] = sum_n x[n] x[n+k] for k = 0..p, straight from the definition."""
    N = frames.shape[-1]
    return np.stack([np.sum(frames[..., :N - k] * frames[..., k:], axis=-1) for k in range(p + 1)], axis=-1)


def center_clip(frames: np.ndarray, ratio: float) -> np.ndarray:
    """Sondhi centre clipping: removes formant structure, keeps the pitch pulses."""
    c = ratio * np.max(np.abs(frames), axis=-1, keepdims=True)
    return np.where(frames > c, frames - c, np.where(frames < -c, frames + c, 0.0))


def pitch_autocorr(frames: np.ndarray, fs: int, f0_min=60.0, f0_max=400.0,
                   clip_ratio=0.3, voicing_thr=0.35):
    """F0 of each frame from the autocorrelation peak. Returns (f0_hz, voicing); f0 = 0 if unvoiced."""
    frames = np.atleast_2d(frames)
    N = frames.shape[-1]
    w = hamming(N)
    r = autocorrelation(center_clip(frames * w, clip_ratio))
    # divide by the window's own autocorrelation, so peak heights are comparable
    rw = autocorrelation(w[None, :])[0]
    r = r / np.maximum(rw, 1e-3 * rw[0])
    r0 = r[:, :1]
    rn = np.where(r0 > 1e-12, r / np.maximum(r0, 1e-12), 0.0)
    lo, hi = int(fs / f0_max), min(int(fs / f0_min), N - 2)
    # a pitch candidate must be a real peak inside the allowed lag range
    mid = rn[:, lo:hi + 1]
    is_peak = (mid > rn[:, lo - 1:hi]) & (mid >= rn[:, lo + 1:hi + 2])
    seg = np.where(is_peak, mid, -np.inf)
    k = np.argmax(seg, axis=1)
    strength = np.maximum(seg[np.arange(len(seg)), k], 0.0)
    lag = (k + lo).astype(float)
    # fit a parabola around the peak for a finer lag estimate
    i = k + lo
    ok = (i > 0) & (i < N - 1)
    a, b, c = rn[np.arange(len(rn)), i - 1], rn[np.arange(len(rn)), i], rn[np.arange(len(rn)), np.minimum(i + 1, N - 1)]
    den = a - 2 * b + c
    delta = np.where(ok & (np.abs(den) > 1e-12), 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    lag = lag + np.clip(delta, -0.5, 0.5)
    f0 = np.where(strength >= voicing_thr, fs / lag, 0.0)
    return f0, strength
