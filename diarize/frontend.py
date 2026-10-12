"""Streaming front-end: microphone samples in, per-frame features out.

    mic -> low-pass FIR -> down-sample to 16 kHz -> NLMS noise canceller, then three branches:
      1. pre-emphasis -> Hamming -> autocorrelation -> AR model -> spectral envelope
      2. band-pass 60-900 Hz -> centre clipping -> autocorrelation -> F0
      3. high-pass 70 Hz -> frame energy (for the VAD); this is also the audio that gets saved
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .sparse import envelope
from .config import Config


class AudioRing:
    """Keeps the last `capacity` samples; indexed by absolute sample number."""

    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.total = 0

    def write(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if len(x) >= self.capacity:
            x = x[-self.capacity:]
        idx = (self.total + np.arange(len(x))) % self.capacity
        self.buf[idx] = x
        self.total += len(x)

    def read(self, start: int, end: int) -> np.ndarray:
        start = max(start, self.total - self.capacity, 0)
        end = min(end, self.total)
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        return self.buf[np.arange(start, end) % self.capacity].copy()


@dataclass
class Frames:
    index: np.ndarray      # absolute frame numbers
    energy_db: np.ndarray  # frame energy (dBFS)
    feat: np.ndarray       # (n, d)  r[1..p]/r[0] (AR model) or MFCC c1..c12
    f0: np.ndarray         # Hz, 0 = unvoiced
    voicing: np.ndarray    # autocorrelation peak height

    def __len__(self):
        return len(self.index)


class _Framer:
    """Accumulates samples and hands out complete overlapping frames."""

    def __init__(self, win: int, hop: int):
        self.win, self.hop = win, hop
        self.buf = np.zeros(0)

    def push(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x])
        fr = dsp.frame_signal(self.buf, self.win, self.hop)
        if len(fr):
            self.buf = self.buf[len(fr) * self.hop:]
        return fr


class FrontEnd:
    def __init__(self, cfg: Config, input_sr: int | None = None):
        self.cfg = cfg
        sr_in = input_sr or cfg.input_sr
        fs = cfg.target_sr
        h = dsp.kaiser_lowpass(cfg.aa_passband_hz, cfg.aa_stopband_hz, cfg.aa_ripple, sr_in) if cfg.aa_window == "kaiser" else None
        self.decim = dsp.Decimator(sr_in, fs, cfg.aa_numtaps, cfg.aa_cutoff_ratio, h)
        self.anc = dsp.AdaptiveNoiseCanceller(cfg.anc_taps, int(cfg.anc_delay_s * fs), cfg.anc_mu, fs,
                                              cfg.anc_gate_db) if cfg.anc_taps else None
        self.pre = dsp.PreEmphasis(cfg.preemph)
        self.bp_speech = dsp.Butterworth(cfg.bp_order, cfg.speech_band, fs, "bandpass")
        self.bp_pitch = dsp.Butterworth(cfg.bp_order, cfg.pitch_band, fs, "bandpass")
        self.hp_audio = dsp.Butterworth(cfg.bp_order, cfg.audio_hp, fs, "highpass")
        self.fr_speech = _Framer(cfg.win_len, cfg.hop_len)
        self.fr_pitch = _Framer(cfg.win_len, cfg.hop_len)
        self.fr_audio = _Framer(cfg.win_len, cfg.hop_len)
        self.window = dsp.hamming(cfg.win_len)
        self.melfb = dsp.mel_filterbank(cfg.n_mels, cfg.n_fft, fs, cfg.fmin, cfg.fmax)
        self.f0_med = dsp.MedianFilter(cfg.f0_median) if cfg.f0_median > 1 else None
        self.ring = AudioRing(int(cfg.ring_seconds * fs))
        self.n_frames = 0

    def process(self, chunk: np.ndarray) -> Frames:
        cfg = self.cfg
        x16 = self.decim.process(np.asarray(chunk, dtype=np.float64))
        if self.anc:
            x16 = self.anc.process(x16)
        ar = cfg.speaker_model != "gauss"
        x_spk = self.pre.process(x16) if ar else self.bp_speech.process(self.pre.process(x16))
        x_pitch = self.bp_pitch.process(x16)
        x_audio = self.hp_audio.process(x16)
        self.ring.write(x_audio)

        f_spk = self.fr_speech.push(x_spk)
        f_pitch = self.fr_pitch.push(x_pitch)
        f_audio = self.fr_audio.push(x_audio)
        n = len(f_spk)  # all three framers advance in lock-step
        idx = np.arange(self.n_frames, self.n_frames + n)
        self.n_frames += n
        if n == 0:
            return Frames(idx, np.zeros(0), np.zeros((0, cfg.feature_dims)), np.zeros(0), np.zeros(0))

        if ar:
            r = dsp.autocorr_lags(f_spk * self.window, cfg.ar_order)
            feat = r[:, 1:] / np.maximum(r[:, :1], 1e-12)                 # gain-normalised r[1..p]
            if cfg.speaker_model == "dict":                               # + log spectral envelope for the dictionary
                feat = np.hstack([feat, envelope(feat, cfg.dict_nfreq)])
        else:
            P = dsp.power_spectrum(f_spk * self.window, cfg.n_fft)
            feat = dsp.mfcc_from_power(P, self.melfb, cfg.n_mfcc)[:, 1:]  # drop c0 (loudness)
        f0, voicing = dsp.pitch_autocorr(f_pitch, cfg.target_sr, cfg.f0_min, cfg.f0_max,
                                         cfg.clip_ratio, cfg.voicing_thr)
        if self.f0_med:
            f0 = self.f0_med.process(f0)
        energy_db = 10 * np.log10(np.mean(f_audio ** 2, axis=1) + 1e-12)
        return Frames(idx, energy_db, feat, f0, voicing)
