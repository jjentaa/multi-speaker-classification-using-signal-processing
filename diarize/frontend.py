"""Streaming front-end: raw microphone samples -> per-frame features.

    mic (48 kHz) -> FIR low-pass -> downsample x3 -> 16 kHz -> NLMS noise canceller  x16
        x16 -> pre-emphasis -> Butterworth band-pass 80-7000 -> Hamming -> DFT -> |X|^2 -> mel -> MFCC
        x16 -> Butterworth band-pass 60-900 -> Hamming -> autocorrelation -> F0
        x16 -> Butterworth high-pass 70 -> frame energy (VAD) + audio kept for output files
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
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
    mfcc: np.ndarray       # (n, n_mfcc-1)   c1..c12
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
        self.decim = dsp.Decimator(sr_in, fs, cfg.aa_numtaps, cfg.aa_cutoff_ratio)
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
        self.ring = AudioRing(int(cfg.ring_seconds * fs))
        self.n_frames = 0

    def process(self, chunk: np.ndarray) -> Frames:
        cfg = self.cfg
        x16 = self.decim.process(np.asarray(chunk, dtype=np.float64))
        if self.anc:
            x16 = self.anc.process(x16)
        x_spk = self.bp_speech.process(self.pre.process(x16))
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
            d = cfg.n_mfcc - 1
            return Frames(idx, np.zeros(0), np.zeros((0, d)), np.zeros(0), np.zeros(0))

        P = dsp.power_spectrum(f_spk * self.window, cfg.n_fft)
        mfcc = dsp.mfcc_from_power(P, self.melfb, cfg.n_mfcc)[:, 1:]  # drop c0 (loudness)
        f0, voicing = dsp.pitch_autocorr(f_pitch, cfg.target_sr, cfg.f0_min, cfg.f0_max,
                                         cfg.clip_ratio, cfg.voicing_thr)
        energy_db = 10 * np.log10(np.mean(f_audio ** 2, axis=1) + 1e-12)
        return Frames(idx, energy_db, mfcc, f0, voicing)
