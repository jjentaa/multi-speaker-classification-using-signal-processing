"""Scores.

DER   = (missed speech + false alarm + wrong speaker) / total speech time
F1    = per speaker, on 10 ms frames, after matching system labels to true speakers
delay = how long after a segment ends the system outputs it
"""
from __future__ import annotations

import warnings

import numpy as np
from pyannote.core import Annotation, Segment as PSeg
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.optimize import linear_sum_assignment

Turns = list[tuple[float, float, str]]


def to_annotation(turns: Turns, uri: str = "session") -> Annotation:
    ann = Annotation(uri=uri)
    for i, (s, e, lab) in enumerate(turns):
        if e > s:
            ann[PSeg(s, e), i] = lab
    return ann


def der(ref: Turns, hyp: Turns, collar: float = 0.25) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    with warnings.catch_warnings():  # hide a harmless pyannote warning
        warnings.simplefilter("ignore")
        r = metric(to_annotation(ref), to_annotation(hyp), detailed=True)
    total = r["total"] or 1.0
    return {
        "DER": r["diarization error rate"],
        "false_alarm": r["false alarm"] / total,
        "missed": r["missed detection"] / total,
        "confusion": r["confusion"] / total,
        "total_ref_s": r["total"],
        "collar": collar,
    }


def _frame_labels(turns: Turns, labels: list[str], n: int, step: float) -> np.ndarray:
    lab = np.full(n, -1)
    idx = {l: k for k, l in enumerate(labels)}
    for s, e, l in turns:
        lab[int(round(s / step)):int(round(e / step))] = idx[l]
    return lab


def speaker_f1(ref: Turns, hyp: Turns, step: float = 0.01) -> dict:
    ref_labels = sorted({t[2] for t in ref})
    hyp_labels = sorted({t[2] for t in hyp})
    T = max([t[1] for t in ref + hyp] + [0.0])
    n = int(np.ceil(T / step)) + 1
    r = _frame_labels(ref, ref_labels, n, step)
    h = _frame_labels(hyp, hyp_labels, n, step)
    # match each system label to the true speaker it overlaps most
    C = np.zeros((len(ref_labels), len(hyp_labels)))
    both = (r >= 0) & (h >= 0)
    np.add.at(C, (r[both], h[both]), 1)
    ri, hi = linear_sum_assignment(-C) if C.size else ([], [])
    mapping = {hyp_labels[j]: ref_labels[i] for i, j in zip(ri, hi)}
    h_mapped = np.full(n, -1)
    for j, hl in enumerate(hyp_labels):
        if hl in mapping:
            h_mapped[h == j] = ref_labels.index(mapping[hl])
        else:
            h_mapped[h == j] = -2  # extra speaker: counted as wrong
    per = {}
    for k, rl in enumerate(ref_labels):
        tp = np.sum((r == k) & (h_mapped == k))
        prec = tp / max(np.sum(h_mapped == k), 1)
        rec = tp / max(np.sum(r == k), 1)
        per[rl] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12)}
    macro = float(np.mean([v["f1"] for v in per.values()])) if per else 0.0
    # accuracy on frames where both say there is speech
    acc = float(np.mean(h_mapped[both] == r[both])) if both.any() else 0.0
    return {"macro_f1": macro, "per_speaker": per, "mapping": mapping, "speech_frame_accuracy": acc,
            "n_ref_speakers": len(ref_labels), "n_hyp_speakers": len(hyp_labels)}


def latency(latencies: list[float]) -> dict:
    a = np.asarray(latencies, dtype=float)
    if a.size == 0:
        return {}
    return {"mean_s": float(a.mean()), "median_s": float(np.median(a)),
            "p95_s": float(np.percentile(a, 95)), "max_s": float(a.max())}


def evaluate(ref: Turns, hyp: Turns, latencies: list[float] | None = None, collar: float = 0.25) -> dict:
    out = {"der": der(ref, hyp, collar), "f1": speaker_f1(ref, hyp)}
    if latencies is not None:
        out["latency"] = latency(latencies)
    return out


def format_report(res: dict) -> str:
    d, f = res["der"], res["f1"]
    lines = [
        f"DER (collar {d['collar']}s)     : {d['DER'] * 100:6.2f} %"
        f"   [miss {d['missed'] * 100:.2f} | FA {d['false_alarm'] * 100:.2f} | confusion {d['confusion'] * 100:.2f}]",
        f"Speaker F1 (macro)      : {f['macro_f1'] * 100:6.2f} %   (ref {f['n_ref_speakers']} spk, found {f['n_hyp_speakers']})",
    ]
    for spk, v in f["per_speaker"].items():
        lines.append(f"    {spk:<12} P {v['precision'] * 100:5.1f}  R {v['recall'] * 100:5.1f}  F1 {v['f1'] * 100:5.1f}")
    lines.append(f"Label accuracy (speech) : {f['speech_frame_accuracy'] * 100:6.2f} %")
    if res.get("latency"):
        l = res["latency"]
        lines.append(f"Time delay              : mean {l['mean_s']:.2f}s  median {l['median_s']:.2f}s  "
                     f"p95 {l['p95_s']:.2f}s  max {l['max_s']:.2f}s")
    return "\n".join(lines)
