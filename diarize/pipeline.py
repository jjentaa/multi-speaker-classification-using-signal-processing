"""Real-time diarization: push audio chunks in, get finished speaker segments out.

The "window" that cuts the audio is a small state machine running on 10 ms
frames:

  SILENCE --(VAD speech)--> SPEECH
  SPEECH  --(pause >= min_silence)---------> cut, emit previous speaker, SILENCE
  SPEECH  --(dBIC peak between the last two 1 s windows)--> cut at the peak,
             emit previous speaker, keep going with the new one
  SPEECH  --(segment longer than max_segment)--> cut

Every emitted segment is labelled by the online clustering, cleaned for ASR
and (optionally) written to disk immediately -- the previous speaker's audio
is usable while the next one is still talking.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .cleaner import clean_segment
from .cluster import OnlineSpeakerClustering, agglomerative_bic
from .config import Config
from .frontend import FrontEnd, Frames
from .gauss import GaussStats, SpeakerStats, delta_bic
from .vad import EnergyVAD


@dataclass
class Segment:
    id: int
    speaker: int
    start: float
    end: float
    confidence: float
    f0_hz: float
    reason: str               # pause | change | max | flush
    emitted_at: float         # stream time (s) when the segment was released
    latency_s: float          # emitted_at - end  (+ processing time)
    file: str | None = None
    stats: SpeakerStats | None = field(default=None, repr=False)
    audio: np.ndarray | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"SPK_{self.speaker + 1}" if self.speaker >= 0 else "UNKNOWN"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("stats"), d.pop("audio")
        d["label"] = self.label
        d["duration"] = round(self.end - self.start, 3)
        for k in ("start", "end", "confidence", "f0_hz", "emitted_at", "latency_s"):
            d[k] = round(float(d[k]), 3)
        return d


class StreamingDiarizer:
    def __init__(self, cfg: Config | None = None, input_sr: int | None = None,
                 out_dir: str | None = None, keep_trace: bool = False):
        self.cfg = cfg = cfg or Config()
        self.input_sr = input_sr or cfg.input_sr
        self.fe = FrontEnd(cfg, self.input_sr)
        self.vad = EnergyVAD(cfg)
        self.clu = OnlineSpeakerClustering(cfg)
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(os.path.join(out_dir, "segments"), exist_ok=True)

        fr = cfg.frame_rate
        self.min_sil = int(round(cfg.min_silence_s * fr))
        self.max_seg = int(round(cfg.max_segment_s * fr))
        self.bic_W = int(round(cfg.bic_win_s * fr))
        self.bic_patience = int(round(cfg.bic_patience_s * fr))
        self.noise_keep = int(cfg.noise_memory_s * cfg.target_sr)

        self.in_speech = False
        self.seg_start = 0            # absolute frame index of the open segment
        self.last_speech = -1
        self._cut_start = False       # True if the open segment began at a speaker-change cut
        self._reset_buffers()
        self.noise = np.zeros(0, dtype=np.float32)
        self.samples_in = 0
        self.segments: list[Segment] = []
        self._bic_best: tuple[float, int] | None = None
        self._next_bic = 0
        self._proc_time = 0.0

        self.keep_trace = keep_trace
        self.trace = {"energy_db": [], "vad": [], "f0": [], "voicing": [], "noise_db": [],
                      "bic_t": [], "bic": []}

    # ------------------------------------------------------------------ utils
    def _reset_buffers(self):
        d = self.cfg.n_mfcc - 1
        self.buf_mfcc = np.zeros((0, d))
        self.buf_f0 = np.zeros(0)
        self.buf_vad = np.zeros(0, dtype=bool)

    def _t(self, frame: float) -> float:
        """Centre time (s) of an absolute frame index."""
        c = self.cfg
        return (frame * c.hop_len + c.win_len / 2) / c.target_sr

    @property
    def stream_time(self) -> float:
        return self.samples_in / self.input_sr

    # ------------------------------------------------------------------ API
    def push(self, chunk: np.ndarray) -> list[Segment]:
        t0 = time.perf_counter()
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        self.samples_in += len(chunk)
        fr = self.fe.process(chunk)
        vad = self.vad.process(fr.energy_db)
        out = self._step(fr, vad)
        self._proc_time = time.perf_counter() - t0
        for s in out:
            s.latency_s += self._proc_time
        return out

    def flush(self) -> list[Segment]:
        out = []
        if self.in_speech and self.last_speech >= self.seg_start:
            out += self._close(self.seg_start, self.last_speech + 1, "flush")
        self.in_speech = False
        self._reset_buffers()
        return out

    # ------------------------------------------------------------------ core
    def _step(self, fr: Frames, vad: np.ndarray) -> list[Segment]:
        cfg, out = self.cfg, []
        if self.keep_trace:
            tr = self.trace
            tr["energy_db"] += list(fr.energy_db)
            tr["vad"] += list(vad)
            tr["f0"] += list(fr.f0)
            tr["voicing"] += list(fr.voicing)
            tr["noise_db"] += [self.vad.noise_db or -100.0] * len(fr)

        for j in range(len(fr)):
            i = int(fr.index[j])
            sp = bool(vad[j])
            if not sp:
                self._add_noise(i)
            if not self.in_speech:
                if not sp:
                    continue
                # speech onset (VAD needed a few frames, start a bit earlier)
                self.in_speech = True
                self.seg_start = i
                self._cut_start = False
                self.last_speech = i
                self._reset_buffers()
                self._bic_best, self._next_bic = None, i + 2 * self.bic_W

            self.buf_mfcc = np.vstack([self.buf_mfcc, fr.mfcc[j:j + 1]])
            self.buf_f0 = np.append(self.buf_f0, fr.f0[j])
            self.buf_vad = np.append(self.buf_vad, sp)
            if sp:
                self.last_speech = i

            if i - self.last_speech >= self.min_sil:              # pause -> cut
                out += self._close(self.seg_start, self.last_speech + 1, "pause")
                self.in_speech = False
                self._reset_buffers()
                continue
            if i + 1 - self.seg_start >= self.max_seg:            # too long -> cut
                out += self._close(self.seg_start, i + 1, "max")
                self._restart_at(i + 1)
                continue
            if i >= self._next_bic:                               # speaker change?
                self._next_bic = i + cfg.bic_step_frames
                c = self._bic_check(i)
                if c is not None:
                    out += self._close(self.seg_start, c, "change")
                    self._restart_at(c)
        return out

    def _restart_at(self, frame: int):
        """Drop buffered frames before `frame`; the open segment now starts there."""
        k = frame - self.seg_start
        self.buf_mfcc, self.buf_f0, self.buf_vad = self.buf_mfcc[k:], self.buf_f0[k:], self.buf_vad[k:]
        self.seg_start = frame
        self._cut_start = True
        self._bic_best = None
        self._next_bic = frame + 2 * self.bic_W

    def _bic_check(self, i: int) -> int | None:
        """dBIC between [s-W, s) and [s, i] with s = i+1-W. Returns a cut frame or None."""
        W, cfg = self.bic_W, self.cfg
        end = i + 1 - self.seg_start          # buffer length
        s = end - W
        if s - W < 0:
            return None
        L, R = slice(s - W, s), slice(s, end)
        mL, mR = self.buf_vad[L], self.buf_vad[R]
        if mL.sum() < cfg.bic_min_speech or mR.sum() < cfg.bic_min_speech:
            d = -np.inf
        else:
            a = SpeakerStats.from_frames(self.buf_mfcc[L][mL], self.buf_f0[L][mL])
            b = SpeakerStats.from_frames(self.buf_mfcc[R][mR], self.buf_f0[R][mR])
            d = delta_bic(a.mfcc, b.mfcc, cfg.bic_lambda)
            if a.logf0.n >= 10 and b.logf0.n >= 10:
                d += cfg.pitch_weight * delta_bic(a.logf0, b.logf0, cfg.bic_lambda)
        split = self.seg_start + s
        if self.keep_trace:
            self.trace["bic_t"].append(self._t(split))
            self.trace["bic"].append(d if np.isfinite(d) else np.nan)
        if d > 0 and (self._bic_best is None or d > self._bic_best[0]):
            self._bic_best = (d, split)
        if self._bic_best is not None and split - self._bic_best[1] >= self.bic_patience:
            cut = self._bic_best[1]
            self._bic_best = None
            return cut
        return None

    def _add_noise(self, frame: int):
        c = self.cfg
        a = frame * c.hop_len
        self.noise = np.concatenate([self.noise, self.fe.ring.read(a, a + c.hop_len)])[-self.noise_keep:]

    def _close(self, f_start: int, f_end: int, reason: str) -> list[Segment]:
        cfg = self.cfg
        k0, k1 = f_start - self.seg_start, f_end - self.seg_start
        mask = self.buf_vad[k0:k1]
        n_speech = int(mask.sum())
        start = self._t(f_start) - cfg.hop_len / cfg.target_sr / 2
        end = self._t(f_end - 1) + cfg.hop_len / cfg.target_sr / 2
        if end - start < cfg.min_segment_s or n_speech < 10:
            return []
        stats = SpeakerStats.from_frames(self.buf_mfcc[k0:k1][mask], self.buf_f0[k0:k1][mask])
        spk, conf, _ = self.clu.assign(stats, n_speech / cfg.frame_rate)

        fs = cfg.target_sr
        pad = int(cfg.pad_s * fs)
        # no padding across a speaker-change cut: it would leak the other voice
        pad_l = 0 if self._cut_start else pad
        pad_r = 0 if reason == "change" else pad
        raw = self.fe.ring.read(int(start * fs) - pad_l, int(end * fs) + pad_r)
        audio = clean_segment(raw, fs, self.noise, cfg)

        seg = Segment(id=len(self.segments), speaker=spk, start=start, end=end, confidence=conf,
                      f0_hz=stats.f0_hz, reason=reason, emitted_at=self.stream_time,
                      latency_s=self.stream_time - end, stats=stats, audio=audio)
        if self.out_dir:
            name = f"{seg.id:04d}_{seg.label}_{start:08.2f}-{end:08.2f}.wav"
            seg.file = os.path.join("segments", name)
            import soundfile as sf  # only needed when writing files (not available in the browser build)
            sf.write(os.path.join(self.out_dir, seg.file), audio, fs, subtype="PCM_16")
        self.segments.append(seg)
        return [seg]

    # ------------------------------------------------------------------ offline
    def refine_labels(self) -> list[int]:
        """Re-cluster all segments at once (agglomerative BIC). Used only for comparison."""
        stats = [s.stats for s in self.segments]
        return list(agglomerative_bic(stats, self.cfg)) if stats else []
