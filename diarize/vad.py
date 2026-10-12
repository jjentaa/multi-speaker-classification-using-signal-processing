"""Energy-based voice activity detection with an adaptive noise floor."""
from __future__ import annotations

import numpy as np

from .config import Config


class EnergyVAD:
    """A frame is speech if its energy is `margin` dB above the noise floor.

    The noise floor follows the energy down quickly and up slowly. A median filter
    then removes one-frame clicks and dropouts from the decision.
    
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.noise_db: float | None = None
        self.run = 0
        self.hist = np.zeros(max(cfg.vad_median - 1, 0), dtype=bool)

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
            out[i] = loud if cfg.vad_median > 1 else self.run >= cfg.vad_onset_frames
        if cfg.vad_median > 1 and len(out):
            buf = np.concatenate([self.hist, out])
            self.hist = buf[len(buf) - (cfg.vad_median - 1):]
            out = np.median(np.lib.stride_tricks.sliding_window_view(buf, cfg.vad_median), axis=1) > 0.5
        return out
