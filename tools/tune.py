"""Grid search over Config parameters on a set of labelled files (mean DER).

    python tools/tune.py data/conv2 data/conv3 data/conv4 --grid ar_change_thr=0.9,1.1,1.3 ar_clu_thr=0.18,0.22,0.26
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarize.config import Config  # noqa: E402
from diarize.evaluate import der, speaker_f1  # noqa: E402
from diarize.io import load_audio, read_rttm  # noqa: E402
from diarize.pipeline import StreamingDiarizer  # noqa: E402

warnings.filterwarnings("ignore")


def run(cfg: Config, x, fs, chunk=0.05):
    d = StreamingDiarizer(cfg, input_sr=fs)
    n = int(fs * chunk)
    for i in range(0, len(x), n):
        d.push(x[i:i + n])
    d.flush()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="path prefixes: <p>.wav + <p>.rttm")
    ap.add_argument("--grid", nargs="*", default=[], help="name=v1,v2,...")
    ap.add_argument("--collar", type=float, default=0.25)
    a = ap.parse_args()
    data = [(load_audio(p + ".wav"), read_rttm(p + ".rttm")) for p in a.sets]
    names = [g.split("=")[0] for g in a.grid]
    values = [[float(v) for v in g.split("=")[1].split(",")] for g in a.grid]
    rows = []
    for combo in itertools.product(*values) if values else [()]:
        cfg = Config()
        for k, v in zip(names, combo):
            setattr(cfg, k, type(getattr(cfg, k))(v))
        ders, f1s, nspk = [], [], []
        for (x, fs), ref in data:
            d = run(cfg, x, fs)
            hyp = [(s.start, s.end, s.label) for s in d.segments]
            ders.append(der(ref, hyp, a.collar)["DER"])
            f = speaker_f1(ref, hyp)
            f1s.append(f["macro_f1"])
            nspk.append(f"{f['n_hyp_speakers']}/{f['n_ref_speakers']}")
        rows.append((np.mean(ders), combo, ders, np.mean(f1s), nspk))
        print(f"{dict(zip(names, combo))}  DER {np.mean(ders) * 100:5.1f}%  "
              f"[{' '.join(f'{v * 100:.1f}' for v in ders)}]  F1 {np.mean(f1s) * 100:.1f}%  spk {' '.join(nspk)}", flush=True)
    best = min(rows, key=lambda r: r[0])
    print(f"\nbest: {dict(zip(names, best[1]))}  DER {best[0] * 100:.1f}%")


if __name__ == "__main__":
    main()
