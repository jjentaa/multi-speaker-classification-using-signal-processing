"""Reading audio and writing / reading results (JSON, RTTM)."""
from __future__ import annotations

import json
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal


def load_audio(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float audio. If `sr` is given and differs, resample (polyphase)."""
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr and fs != sr:
        g = gcd(sr, fs)
        x = signal.resample_poly(x, sr // g, fs // g)
        fs = sr
    return x, fs


def write_json(path: str, segments, meta: dict):
    data = {"meta": meta, "segments": [s.to_dict() for s in segments]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_rttm(path: str, segments, uri: str = "session", labels=None):
    with open(path, "w") as f:
        for i, s in enumerate(segments):
            lab = f"SPK_{labels[i] + 1}" if labels is not None else s.label
            f.write(f"SPEAKER {uri} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {lab} <NA> <NA>\n")


def read_rttm(path: str) -> list[tuple[float, float, str]]:
    """[(start, end, speaker), ...]"""
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                st, du = float(p[3]), float(p[4])
                out.append((st, st + du, p[7]))
    return out


def read_json_segments(path: str) -> list[tuple[float, float, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(s["start"], s["end"], s["label"]) for s in data["segments"]]


def load_turns(path: str) -> list[tuple[float, float, str]]:
    return read_json_segments(path) if os.path.splitext(path)[1] == ".json" else read_rttm(path)
