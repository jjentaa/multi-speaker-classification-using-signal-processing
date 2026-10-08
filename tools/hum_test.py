"""DER with the adaptive noise canceller off / on, with mains hum added to labelled files.

The hum is 50 Hz plus harmonics up to 600 Hz (amplitude 1/k); its level is given
as a speech-to-hum ratio in dB.

    python tools/hum_test.py data/test2 data/test3 data/test4 --snr 20 10 0
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarize.config import Config  # noqa: E402
from diarize.evaluate import der, speaker_f1  # noqa: E402
from diarize.io import load_audio, read_rttm  # noqa: E402
from tune import run  # noqa: E402

warnings.filterwarnings("ignore")


def add_hum(x, fs, ref, snr_db, f0=50.0, harmonics=12, seed=1):
    t = np.arange(len(x)) / fs
    rng = np.random.default_rng(seed)
    hum = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, harmonics + 1))
    speech_pow = np.mean(np.concatenate([x[int(s * fs):int(e * fs)] for s, e, _ in ref]) ** 2)
    return x + hum * np.sqrt(speech_pow / np.mean(hum ** 2) / 10 ** (snr_db / 10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="path prefixes: <p>.wav + <p>.rttm")
    ap.add_argument("--snr", nargs="*", type=float, default=[20.0, 10.0, 0.0], help="speech-to-hum ratios in dB")
    ap.add_argument("--collar", type=float, default=0.25)
    a = ap.parse_args()
    taps = Config().anc_taps
    for p in a.sets:
        (x, fs), ref = load_audio(p + ".wav"), read_rttm(p + ".rttm")
        for snr in [None] + a.snr:
            y = x if snr is None else add_hum(x, fs, ref, snr)
            cols = []
            for on in (False, True):
                cfg = Config()
                cfg.anc_taps = taps if on else 0
                d = run(cfg, y, fs)
                hyp = [(s.start, s.end, s.label) for s in d.segments]
                f = speaker_f1(ref, hyp)
                cols.append(f"{'on ' if on else 'off'} DER {der(ref, hyp, a.collar)['DER'] * 100:5.1f}%  "
                            f"F1 {f['macro_f1'] * 100:5.1f}%  spk {f['n_hyp_speakers']}/{f['n_ref_speakers']}")
            print(f"{os.path.basename(p)}  hum {'none' if snr is None else f'{snr:.0f} dB':>5} | " + " | ".join(cols), flush=True)


if __name__ == "__main__":
    main()
