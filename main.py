"""Real-time multi-speaker diarization -- command line.

    python main.py live  --out runs/live                 # microphone, Ctrl+C to stop
    python main.py file  data/conv3.wav --ref data/conv3.rttm --out runs/conv3 --plot
    python main.py eval  --ref data/conv3.rttm --hyp runs/conv3/result.json
    python main.py devices
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import sys
import time

import numpy as np
import soundfile as sf

from diarize.config import Config
from diarize.evaluate import evaluate, format_report
from diarize.io import load_audio, load_turns, write_json, write_rttm
from diarize.pipeline import Segment, StreamingDiarizer


def print_segment(s: Segment):
    print(f"[{s.start:7.2f} - {s.end:7.2f}] {s.label:<8} conf {s.confidence:.2f}  F0 {s.f0_hz:5.0f} Hz  "
          f"cut:{s.reason:<6} delay {s.latency_s:.2f}s  {s.file or ''}", flush=True)


def save_results(d: StreamingDiarizer, out: str, meta: dict, refine: bool):
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "result.json"), d.segments, meta)
    write_rttm(os.path.join(out, "result.rttm"), d.segments, uri=meta.get("uri", "session"))
    if refine and d.segments:
        labels = d.refine_labels()
        write_rttm(os.path.join(out, "result_offline.rttm"), d.segments, uri=meta.get("uri", "session"), labels=labels)
    print(f"\nsaved {len(d.segments)} segments -> {out}/result.json, {out}/result.rttm, {out}/segments/")


def run_eval(d: StreamingDiarizer, ref_path: str, out: str, collar: float, refine: bool):
    ref = load_turns(ref_path)
    hyp = [(s.start, s.end, s.label) for s in d.segments]
    res = evaluate(ref, hyp, [s.latency_s for s in d.segments], collar)
    print("\n=== Evaluation (online / real-time labels) ===")
    print(format_report(res))
    report = {"online": res}
    if refine and d.segments:
        labels = d.refine_labels()
        hyp2 = [(s.start, s.end, f"SPK_{l + 1}") for s, l in zip(d.segments, labels)]
        res2 = evaluate(ref, hyp2, None, collar)
        print("\n=== Evaluation (offline agglomerative re-clustering, for comparison) ===")
        print(format_report(res2))
        report["offline"] = res2
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


# ---------------------------------------------------------------------- file
def cmd_file(a, cfg: Config):
    x, fs = load_audio(a.audio)
    if fs % cfg.target_sr:
        x, fs = load_audio(a.audio, cfg.input_sr)   # e.g. 44.1 kHz -> 48 kHz
    d = StreamingDiarizer(cfg, input_sr=fs, out_dir=a.out, keep_trace=a.plot)
    chunk = int(fs * a.chunk_ms / 1000)
    t0 = time.perf_counter()
    for i in range(0, len(x), chunk):
        for s in d.push(x[i:i + chunk]):
            print_segment(s)
        if a.realtime:
            time.sleep(max(0.0, (i + chunk) / fs - (time.perf_counter() - t0)))
    for s in d.flush():
        print_segment(s)
    wall = time.perf_counter() - t0
    dur = len(x) / fs
    print(f"\nprocessed {dur:.1f} s of audio in {wall:.1f} s  (real-time factor {wall / dur:.3f})")
    uri = os.path.splitext(os.path.basename(a.audio))[0]
    save_results(d, a.out, {"source": a.audio, "uri": uri, "duration_s": dur, "rtf": wall / dur,
                            "n_speakers": len(d.clu.speakers)}, a.refine)
    if a.ref:
        run_eval(d, a.ref, a.out, a.collar, a.refine)
    if a.plot:
        from diarize.plots import plot_session
        p = os.path.join(a.out, "session.png")
        plot_session(x, fs, d.trace, cfg, d.segments, load_turns(a.ref) if a.ref else None, p)
        print(f"plots -> {p}")


# ---------------------------------------------------------------------- live
def cmd_live(a, cfg: Config):
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    recorded = []

    def callback(indata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    sr = a.sr or cfg.input_sr
    d = StreamingDiarizer(cfg, input_sr=sr, out_dir=a.out)
    block = int(sr * a.chunk_ms / 1000)
    print(f"listening on device {a.device if a.device is not None else 'default'} @ {sr} Hz "
          f"(block {a.chunk_ms} ms) -- Ctrl+C to stop\n")
    t_start = time.time()
    warned = False
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block,
                            device=a.device, callback=callback):
            while a.duration is None or time.time() - t_start < a.duration:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    if not recorded and time.time() - t_start > 3 and not warned:
                        warned = True
                        print("no audio from the microphone yet -- on macOS allow your terminal app in "
                              "System Settings > Privacy & Security > Microphone, or pick a device with --device",
                              file=sys.stderr)
                    continue
                recorded.append(chunk)
                for s in d.push(chunk):
                    # add the time the block spent waiting in the queue / sound card buffer
                    s.latency_s += block / sr + q.qsize() * block / sr
                    print_segment(s)
    except KeyboardInterrupt:
        pass
    for s in d.flush():
        print_segment(s)
    x = np.concatenate(recorded) if recorded else np.zeros(0)
    os.makedirs(a.out, exist_ok=True)
    sf.write(os.path.join(a.out, "recording.wav"), x, sr, subtype="PCM_16")
    save_results(d, a.out, {"source": "microphone", "uri": "live", "duration_s": len(x) / sr,
                            "n_speakers": len(d.clu.speakers)}, a.refine)
    lat = [s.latency_s for s in d.segments]
    if lat:
        print(f"time delay: mean {np.mean(lat):.2f}s  max {np.max(lat):.2f}s")
    print(f"full recording -> {a.out}/recording.wav (label it and use `main.py eval` to score)")


def cmd_eval(a, _cfg):
    ref, hyp = load_turns(a.ref), load_turns(a.hyp)
    lat = None
    if a.hyp.endswith(".json"):
        with open(a.hyp) as f:
            lat = [s["latency_s"] for s in json.load(f)["segments"]]
    print(format_report(evaluate(ref, hyp, lat, a.collar)))


def cmd_devices(_a, _cfg):
    import sounddevice as sd
    print(sd.query_devices())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="runs/out", help="output folder")
        p.add_argument("--chunk-ms", type=float, default=50.0, help="audio block size pushed per step")
        p.add_argument("--max-speakers", type=int, default=4)
        p.add_argument("--refine", action="store_true", help="also run offline re-clustering at the end")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")

    p = sub.add_parser("file", help="run on an audio file, streamed in small chunks")
    p.add_argument("audio")
    p.add_argument("--ref", help="reference RTTM/JSON -> compute DER, F1, delay")
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--realtime", action="store_true", help="feed chunks at real speed")
    common(p)
    p.set_defaults(fn=cmd_file)

    p = sub.add_parser("live", help="real-time from the microphone")
    p.add_argument("--device", type=lambda v: int(v) if v.isdigit() else v, default=None)
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    common(p)
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("eval", help="score a result against a reference")
    p.add_argument("--ref", required=True)
    p.add_argument("--hyp", required=True)
    p.add_argument("--collar", type=float, default=0.25)
    p.set_defaults(fn=cmd_eval)

    sub.add_parser("devices", help="list audio devices").set_defaults(fn=cmd_devices)

    a = ap.parse_args()
    cfg = Config()
    if hasattr(a, "max_speakers"):
        cfg.max_speakers = a.max_speakers
    for kv in getattr(a, "set", []):
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, tuple(ast.literal_eval(v)) if isinstance(cur, tuple) else type(cur)(v))
    a.fn(a, cfg)


if __name__ == "__main__":
    main()
