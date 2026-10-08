"""Signal-processing building blocks (proposal section 2.1).

Every filter here is *stateful* so that audio can be pushed in arbitrary
chunks (real-time) and the output is identical to filtering the whole signal
at once.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import dct


# ---------------------------------------------------------------------------
# 2.1.2 Low-pass (FIR anti-aliasing) + 2.1.3 Down-sampling
# ---------------------------------------------------------------------------
class Decimator:
    """Anti-aliasing FIR low-pass followed by keeping every M-th sample.

    Down-sampling by M folds everything above fs_new/2 back into the band, so
    the signal must first be low-passed below fs_new/2 (cutoff = ratio*fs_new/2).
    """

    def __init__(self, fs_in: int, fs_out: int, numtaps: int = 97, cutoff_ratio: float = 0.9):
        if fs_in % fs_out:
            raise ValueError(f"fs_in ({fs_in}) must be an integer multiple of fs_out ({fs_out})")
        self.M = fs_in // fs_out
        self.h = signal.firwin(numtaps, cutoff_ratio * (fs_out / 2), fs=fs_in, window="hamming") if self.M > 1 else np.array([1.0])
        self.zi = np.zeros(len(self.h) - 1)
        self.phase = 0  # index (inside the next chunk) of the next sample to keep

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.M == 1:
            return x.astype(np.float64)
        y, self.zi = signal.lfilter(self.h, 1.0, x, zi=self.zi)
        out = y[self.phase::self.M]
        self.phase = (self.phase - len(x)) % self.M
        return out


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
# Adaptive filtering (NLMS)   e[n] = d[n] - h^T x[n]
# ---------------------------------------------------------------------------
class AdaptiveNoiseCanceller:
    """Single-microphone adaptive noise canceller (adaptive line enhancer).

    Same structure as an echo canceller, but with only one microphone there is
    no separate reference, so the reference is the microphone itself delayed
    by D samples:

        d[n] = v[n] + y[n]      microphone = speech + periodic interference
        x[n] = d[n - D]         reference
        y^[n] = h^T x[n]        adaptive FIR estimate of the interference
        e[n] = d[n] - y^[n]     output

    Hum / fan tones are periodic for ever, so they can be predicted from D
    samples ago and are subtracted. The weights follow NLMS,
    h += mu * e[n] * x[n] / (eps + ||x[n]||^2 + L e[n]^2), but only while the output level
    is near its noise floor: voiced speech and reverberation are predictable
    over D too, so adapting during speech would make the filter cancel the
    talker (the same reason an echo canceller freezes during double-talk).
    """

    def __init__(self, taps: int = 512, delay: int = 480, mu: float = 0.02, fs: int = 16000,
                 gate_db: float = 6.0, eps: float = 1e-6):
        self.L, self.mu, self.eps = taps, mu, eps
        self.h = np.zeros(taps)                     # h[k] multiplies d[n - D - (L-1) + k]
        self.hist = np.zeros(taps - 1 + delay)      # microphone samples before the next chunk
        self.a = np.exp(-1.0 / (0.010 * fs))        # 10 ms power smoothing
        self.gate = 10 ** (gate_db / 10)
        self.rise = 10 ** (3.0 / 10 / fs)           # noise floor may rise 3 dB / s
        self.p = 0.0                                # smoothed output power
        self.floor = 0.0                            # output noise floor (power)
        self.warmup = int(0.05 * fs)                # samples left before p is a valid power estimate
        self.adapting = 0                           # samples adapted on (for diagnostics)

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
    """Direct O(N^2) DFT  X[k] = sum_n x[n] e^{-j 2 pi k n / N}.

    Kept as the reference definition; the pipeline uses the FFT (same result,
    O(N log N)) -- see tests/test_dsp.py.
    """
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
    """Linear (not circular) autocorrelation of each frame via the FFT.

    Zero-padding to >= 2N-1 makes the circular correlation computed by the
    DFT equal to the linear one.
    """
    N = frames.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.rfft(frames, n=nfft, axis=-1)
    r = np.fft.irfft(np.abs(X) ** 2, n=nfft, axis=-1)
    return r[..., :N]


def center_clip(frames: np.ndarray, ratio: float) -> np.ndarray:
    """Sondhi centre clipping: removes formant structure, keeps the pitch pulses."""
    c = ratio * np.max(np.abs(frames), axis=-1, keepdims=True)
    return np.where(frames > c, frames - c, np.where(frames < -c, frames + c, 0.0))


def pitch_autocorr(frames: np.ndarray, fs: int, f0_min=60.0, f0_max=400.0,
                   clip_ratio=0.3, voicing_thr=0.35):
    """F0 per frame from the peak of the normalised autocorrelation.

    Returns (f0_hz, voicing_strength); f0 = 0 for unvoiced frames.
    """
    frames = np.atleast_2d(frames)
    N = frames.shape[-1]
    w = hamming(N)
    r = autocorrelation(center_clip(frames * w, clip_ratio))
    # divide by the autocorrelation of the window itself -> unbiased peak heights
    rw = autocorrelation(w[None, :])[0]
    r = r / np.maximum(rw, 1e-3 * rw[0])
    r0 = r[:, :1]
    rn = np.where(r0 > 1e-12, r / np.maximum(r0, 1e-12), 0.0)
    lo, hi = int(fs / f0_max), min(int(fs / f0_min), N - 2)
    # only true local maxima inside [lo, hi] are pitch candidates (no edge peaks)
    mid = rn[:, lo:hi + 1]
    is_peak = (mid > rn[:, lo - 1:hi]) & (mid >= rn[:, lo + 1:hi + 2])
    seg = np.where(is_peak, mid, -np.inf)
    k = np.argmax(seg, axis=1)
    strength = np.maximum(seg[np.arange(len(seg)), k], 0.0)
    lag = (k + lo).astype(float)
    # parabolic interpolation around the peak for sub-sample lag
    i = k + lo
    ok = (i > 0) & (i < N - 1)
    a, b, c = rn[np.arange(len(rn)), i - 1], rn[np.arange(len(rn)), i], rn[np.arange(len(rn)), np.minimum(i + 1, N - 1)]
    den = a - 2 * b + c
    delta = np.where(ok & (np.abs(den) > 1e-12), 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    lag = lag + np.clip(delta, -0.5, 0.5)
    f0 = np.where(strength >= voicing_thr, fs / lag, 0.0)
    return f0, strength
