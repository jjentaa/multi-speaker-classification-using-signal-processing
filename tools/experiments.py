"""Experiments of the report, all on the current pipeline (dictionary speaker model).

    E1  aliasing test: a 10 kHz tone with and without the FIR before keeping every 3rd sample
    E2  mains hum at four levels, NLMS canceller off / on (mean DER over the six synthetic files)
    E3  F0 on / off in the speaker model and the change detector
    E4  VAD median filter on / off
    E6  speaker-change cuts: recall of the true changes and precision of the "change" cuts
    E7  whole pipeline on synthetic, CALLHOME (first 10) and AMI (3 meetings)

    python tools/experiments.py              # everything
    python tools/experiments.py e1 e6        # some of them
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from diarize import dsp  # noqa: E402
from diarize.config import Config  # noqa: E402
from diarize.evaluate import der, speaker_f1  # noqa: E402
from diarize.io import load_audio, read_rttm  # noqa: E402
from hum_test import add_hum  # noqa: E402
from tune import run  # noqa: E402

warnings.filterwarnings("ignore")

SYN = [os.path.join(ROOT, "data", n) for n in ("conv2", "conv3", "conv4", "test2", "test3", "test4")]
AMI = [os.path.join(ROOT, "data", "ami", n) for n in ("ES2004a", "IS1009a", "TS3003a")]
CALLHOME_PARQUET = os.path.join(ROOT, "data", "callhome", "eng-00000.parquet")
OUT = os.path.join(ROOT, "runs", "experiments.json")
FIG = os.path.join(ROOT, "docs", "figures")
NO_F0 = {"dict_pitch_weight": 0.0, "dict_change_pitch": 0.0}
NO_MEDIAN = {"vad_median": 0, "vad_onset_frames": 1}


# ---------------------------------------------------------------------------- data
def callhome_files(n: int = 10) -> list[str]:
    """Write the first n CALLHOME conversations as wav + rttm (once) and return their path prefixes."""
    import pyarrow.parquet as pq
    import soundfile as sf
    out = []
    rows = None
    for i in range(n):
        p = os.path.join(ROOT, "data", "callhome", f"eng-{i:03d}")
        if not (os.path.exists(p + ".wav") and os.path.exists(p + ".rttm")):
            rows = rows or pq.read_table(CALLHOME_PARQUET).slice(0, n).to_pylist()
            r = rows[i]
            x, fs = sf.read(io.BytesIO(r["audio"]["bytes"]))
            sf.write(p + ".wav", x, fs)
            with open(p + ".rttm", "w") as f:
                for s, e, lab in zip(r["timestamps_start"], r["timestamps_end"], r["speakers"]):
                    f.write(f"SPEAKER eng-{i:03d} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")
        out.append(p)
    return out


# ---------------------------------------------------------------------------- one run
def change_eval(ref, segs, tol: float) -> dict:
    """Speaker changes in the reference vs. the boundaries between consecutive output segments.

    A true change is the gap [end of a turn, start of the next turn by another speaker]; a boundary is
    [end of one segment, start of the next]. They match if the two intervals are at most `tol` apart.
    """
    ref = sorted(ref)
    true = [(a[1], b[0], b[0] - a[1]) for a, b in zip(ref, ref[1:]) if a[2] != b[2]]
    segs = sorted(segs)
    bounds = [(p[1], n[0], p[2]) for p, n in zip(segs, segs[1:])]

    def near(a, b):
        return max(0.0, max(a[0], b[0]) - min(a[1], b[1])) <= tol

    found = [any(near(t, b) for b in bounds) for t in true]
    short = [f for f, t in zip(found, true) if t[2] < 0.3]            # no pause the VAD could cut at
    change_cuts = [b for b in bounds if b[2] == "change"]
    good = [any(near(b, t) for t in true) for b in change_cuts]
    return {"n_true": len(true), "found": int(sum(found)), "n_short": len(short), "found_short": int(sum(short)),
            "n_change_cuts": len(change_cuts), "good_change_cuts": int(sum(good))}


def run_one(job: dict) -> dict:
    path, over, hum = job["path"], job.get("over", {}), job.get("hum")
    x, fs = load_audio(path + ".wav", 48000)
    ref = read_rttm(path + ".rttm")
    if hum is not None:
        x = add_hum(x, fs, ref, hum)
    cfg = Config()
    for k, v in over.items():
        setattr(cfg, k, type(getattr(cfg, k))(v))
    t0 = time.perf_counter()
    d = run(cfg, x, fs)
    wall = time.perf_counter() - t0
    hyp = [(s.start, s.end, s.label) for s in d.segments]
    r, f = der(ref, hyp, 0.25), speaker_f1(ref, hyp)
    one = der(ref, [(s, e, "X") for s, e, _ in ref], 0.25)["DER"]     # everything = one speaker, perfect VAD
    lat = [s.latency_s for s in d.segments] or [0.0]
    segs = [(s.start, s.end, s.reason) for s in d.segments]
    return {**job, "name": os.path.basename(path), "DER": r["DER"], "miss": r["missed"], "fa": r["false_alarm"],
            "conf": r["confusion"], "f1": f["macro_f1"], "n_hyp": f["n_hyp_speakers"], "n_ref": f["n_ref_speakers"],
            "one_spk_DER": one, "delay_mean": float(np.mean(lat)), "delay_max": float(np.max(lat)),
            "rtf": wall / (len(x) / fs), "dur_s": len(x) / fs, "n_seg": len(segs), "n_turns": len(ref),
            "change_025": change_eval(ref, segs, 0.25), "change_05": change_eval(ref, segs, 0.5)}


# ---------------------------------------------------------------------------- E1
def e1() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    cfg = Config()
    x, fs = load_audio(os.path.join(ROOT, "data", "conv2.wav"), cfg.input_sr)
    t = np.arange(len(x)) / fs
    a = np.sqrt(2 * np.mean(x ** 2))                       # tone as loud as the whole recording (RMS)
    y = x + a * np.sin(2 * np.pi * 10000 * t)
    h = dsp.kaiser_lowpass(cfg.aa_passband_hz, cfg.aa_stopband_hz, cfg.aa_ripple, fs)
    naive = y[::3]
    filt = dsp.Decimator(fs, cfg.target_sr, h=h).process(y)
    clean = dsp.Decimator(fs, cfg.target_sr, h=h).process(x)
    t16 = np.arange(len(naive)) / cfg.target_sr

    def level(sig, f):                                     # amplitude of the tone at f, in dB re. the 10 kHz tone
        n = min(len(sig), len(t16))
        return 20 * np.log10(2 * np.abs(np.mean(sig[:n] * np.exp(-2j * np.pi * f * t16[:n]))) / a + 1e-15)

    res = {"alias_no_filter_db": level(naive, 6000), "alias_with_filter_db": level(filt, 6000),
           "speech_only_at_6k_db": level(clean, 6000)}

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for sig, name, c, lw, ls in ((naive, "no low-pass: keep every 3rd sample", "#c0392b", 1.2, "-"),
                                 (filt, "Kaiser FIR, then keep every 3rd sample", "#1f5fa6", 2.6, "-"),
                                 (clean, "speech only, no tone (lies on the blue line)", "#222222", 0.8, "--")):
        f, P = signal.welch(sig, cfg.target_sr, nperseg=4096)
        ax.semilogy(f / 1000, P, label=name, color=c, lw=lw, ls=ls)
    ax.axvline(6, color="k", ls=":", lw=1)
    ax.annotate("10 kHz tone folds to 16 − 10 = 6 kHz", xy=(6, ax.get_ylim()[1] / 30), xytext=(6.2, ax.get_ylim()[1] / 30),
                fontsize=10)
    ax.set(title="Aliasing test: 10 kHz tone added to conv2 before down-sampling 48 → 16 kHz", xlabel="kHz",
           xlim=(0, 8))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "01b_alias_test.png"), dpi=130)
    print(f"E1  tone at 6 kHz after down-sampling: no filter {res['alias_no_filter_db']:.1f} dB, "
          f"with FIR {res['alias_with_filter_db']:.1f} dB (speech alone there: {res['speech_only_at_6k_db']:.1f} dB)")
    return res


# ---------------------------------------------------------------------------- reports
def mean(rows, key):
    return 100 * float(np.mean([r[key] for r in rows]))


def pick(results, exp, **kw):
    return [r for r in results if r["exp"] == exp and all(r.get(k) == v for k, v in kw.items())]


def change_sum(rows, key):
    c = [r[key] for r in rows]
    s = {k: sum(x[k] for x in c) for k in c[0]}
    return s


def report(results: list[dict]):
    if pick(results, "e2"):
        print("\nE2  mean DER over the six synthetic files")
        print("    hum            canceller off   canceller on")
        for hum in (None, 20.0, 10.0, 0.0):
            off, on = pick(results, "e2", hum=hum, anc="off"), pick(results, "e2", hum=hum, anc="on")
            name = "no hum" if hum is None else f"{hum:.0f} dB below"
            print(f"    {name:<14} {mean(off, 'DER'):10.1f} %   {mean(on, 'DER'):10.1f} %")
    base = pick(results, "e7", set="syn")
    for exp, label, var in (("e3", "F0", "no F0"), ("e4", "VAD median filter", "no median")):
        alt = pick(results, exp)
        if not alt or not base:
            continue
        print(f"\n{exp.upper()}  {label}: current vs {var} (six synthetic files)")
        for name, rows in (("current", base), (var, alt)):
            ch = change_sum(rows, "change_05")
            print(f"    {name:<10} DER {mean(rows, 'DER'):5.1f} %  (miss {mean(rows, 'miss'):.1f} / fa {mean(rows, 'fa'):.1f}"
                  f" / conf {mean(rows, 'conf'):.1f})  F1 {mean(rows, 'f1'):.1f} %  speakers "
                  f"{' '.join(str(r['n_hyp']) + '/' + str(r['n_ref']) for r in rows)}  segments {sum(r['n_seg'] for r in rows)}"
                  f" (turns {sum(r['n_turns'] for r in rows)})  changes found {ch['found']}/{ch['n_true']}")
    if base:
        print("\nE6  speaker-change cuts (six synthetic files)")
        for key, tol in (("change_025", 0.25), ("change_05", 0.5)):
            s = change_sum(base, key)
            print(f"    ±{tol} s: true changes found {s['found']}/{s['n_true']} ({100 * s['found'] / s['n_true']:.0f} %),"
                  f" of which without a pause {s['found_short']}/{s['n_short']};"
                  f" 'change' cuts correct {s['good_change_cuts']}/{s['n_change_cuts']}"
                  f" ({100 * s['good_change_cuts'] / max(s['n_change_cuts'], 1):.0f} %)")
    for setname, title in (("syn", "synthetic (conv + test)"), ("callhome", "CALLHOME"), ("ami", "AMI")):
        rows = pick(results, "e7", set=setname)
        if not rows:
            continue
        print(f"\nE7  {title}")
        for r in sorted(rows, key=lambda r: r["name"]):
            print(f"    {r['name']:<9} DER {100 * r['DER']:5.1f} %  (miss {100 * r['miss']:4.1f} / fa {100 * r['fa']:4.1f} / "
                  f"conf {100 * r['conf']:4.1f})  F1 {100 * r['f1']:5.1f} %  spk {r['n_hyp']}/{r['n_ref']}  "
                  f"one-speaker DER {100 * r['one_spk_DER']:5.1f} %  delay max {r['delay_max']:.2f} s  RTF {r['rtf']:.3f}")
        print(f"    mean      DER {mean(rows, 'DER'):5.1f} %  F1 {mean(rows, 'f1'):5.1f} %  one-speaker DER "
              f"{mean(rows, 'one_spk_DER'):5.1f} %")


# ---------------------------------------------------------------------------- main
def main():
    want = set(a.lower() for a in sys.argv[1:]) or {"e1", "e2", "e3", "e4", "e6", "e7"}
    results = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f)
        results = [r for r in old.get("runs", []) if r["exp"] not in want and not ({"e3", "e4", "e6"} & want
                                                                                    and r["exp"] == "e7" and r["set"] == "syn")]
    extra = {}
    if "e1" in want:
        extra["e1"] = e1()

    # synthetic baseline, one file at a time so the real-time factor is not disturbed by other runs
    if want & {"e3", "e4", "e6", "e7"}:
        for p in SYN:
            r = run_one({"exp": "e7", "set": "syn", "path": p})
            print(f"  syn {r['name']}: DER {100 * r['DER']:.1f} %  RTF {r['rtf']:.3f}", flush=True)
            results.append(r)

    jobs = []
    if "e2" in want:
        for p in SYN:
            for hum in (None, 20.0, 10.0, 0.0):
                jobs.append({"exp": "e2", "path": p, "hum": hum, "anc": "on"})
                jobs.append({"exp": "e2", "path": p, "hum": hum, "anc": "off", "over": {"anc_taps": 0}})
    if "e3" in want:
        jobs += [{"exp": "e3", "path": p, "over": NO_F0} for p in SYN]
    if "e4" in want:
        jobs += [{"exp": "e4", "path": p, "over": NO_MEDIAN} for p in SYN]
    if "e7" in want:
        jobs += [{"exp": "e7", "set": "ami", "path": p} for p in AMI]
        jobs += [{"exp": "e7", "set": "callhome", "path": p} for p in callhome_files()]
    jobs.sort(key=lambda j: -os.path.getsize(j["path"] + ".wav"))   # longest first

    with Pool(min(8, os.cpu_count() or 1)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            results.append(r)
            print(f"  [{k}/{len(jobs)}] {r['exp']} {r['name']} {r.get('set', '')} hum={r.get('hum')}: "
                  f"DER {100 * r['DER']:.1f} %", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"runs": results, **extra}, f, indent=1, default=str)
    report(results)


if __name__ == "__main__":
    main()
