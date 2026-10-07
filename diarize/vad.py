"""Energy-based voice activity detection with an adaptive noise floor."""
from __future__ import annotations

import numpy as np

from .config import Config


class EnergyVAD:
    """Speech if frame energy exceeds the tracked noise floor by `margin` dB.

    The noise floor follows the energy down quickly and up only slowly, so it
    settles on the background level between words (minimum-statistics idea).
    A few consecutive loud frames are needed before speech starts (onset).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.noise_db: float | None = None
        self.run = 0

    def process(self, energy_db: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        out = np.zeros(len(energy_db), dtype=bool)
        for i, e in enumerate(energy_db):
            if self.noise_db is None:
                self.noise_db = e
            if e < self.noise_db:
                self.noise_db += cfg.noise_fall_rate * (e - self.noise_db)
            else:
                self.noise_db += cfg.noise_rise_rate * (e - self.noise_db)
            loud = e > max(self.noise_db + cfg.vad_margin_db, cfg.vad_abs_floor_db)
            self.run = self.run + 1 if loud else 0
            out[i] = self.run >= cfg.vad_onset_frames
        return out
