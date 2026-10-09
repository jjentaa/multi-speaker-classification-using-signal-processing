// Runs the Python diarizer (the same diarize/ package as the CLI) inside the browser with Pyodide.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
importScripts(PYODIDE + "pyodide.js");

const MODULES = ["__init__", "config", "dsp", "frontend", "gauss", "cluster", "vad", "cleaner", "pipeline"];

const BRIDGE = `
import sys, json, time
sys.path.insert(0, "/app")
import numpy as np
from diarize.config import Config
from diarize.pipeline import StreamingDiarizer

_d = None

def start(sr, anc):
    global _d
    cfg = Config()
    if not anc:
        cfg.anc_taps = 0
    _d = StreamingDiarizer(cfg, input_sr=int(sr), keep_trace=True)

def _pack(segs, dt):
    tr = _d.trace
    meta = {
        "segments": [s.to_dict() for s in segs],
        "energy": [round(float(v), 1) for v in tr["energy_db"]],
        "noise": [round(float(v), 1) for v in tr["noise_db"]],
        "vad": [int(v) for v in tr["vad"]],
        "f0": [round(float(v), 1) for v in tr["f0"]],
        "time": _d.stream_time,
        "proc_ms": dt * 1000.0,
        "n_speakers": len(_d.clu.speakers),
    }
    for v in tr.values():
        v.clear()
    return json.dumps(meta), [np.asarray(s.audio, dtype=np.float32) for s in segs]

def push(buf):
    t0 = time.perf_counter()
    segs = _d.push(np.asarray(buf.to_py(), dtype=np.float64))
    return _pack(segs, time.perf_counter() - t0)

def flush():
    t0 = time.perf_counter()
    return _pack(_d.flush(), time.perf_counter() - t0)
`;

let py = null;

async function init() {
  postMessage({ type: "status", text: "Loading Python runtime…" });
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", text: "Loading numpy and scipy…" });
  await py.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", text: "Loading diarizer…" });
  py.FS.mkdirTree("/app/diarize");
  await Promise.all(MODULES.map(async (m) => {
    const r = await fetch(`/diarize/${m}.py`);
    if (!r.ok) throw new Error(`could not load diarize/${m}.py (${r.status})`);
    py.FS.writeFile(`/app/diarize/${m}.py`, await r.text());
  }));
  py.runPython(BRIDGE);
  postMessage({ type: "ready" });
}

function call(name, ...args) {
  const fn = py.globals.get(name);
  const res = fn(...args);
  fn.destroy();
  if (res === undefined || res === null) return null;
  const [meta, audio] = res.toJs();
  res.destroy();
  return { meta: JSON.parse(meta), audio };
}

function reply(type, r, extra = {}) {
  postMessage({ type, ...r.meta, audio: r.audio, ...extra }, r.audio.map((a) => a.buffer));
}

const ready = init().catch((e) => postMessage({ type: "error", text: String(e) }));

onmessage = async (ev) => {
  await ready;
  const m = ev.data;
  try {
    if (m.type === "start") {
      call("start", m.sampleRate, m.anc);
    } else if (m.type === "push") {
      reply("result", call("push", m.chunk));
    } else if (m.type === "flush") {
      reply("result", call("flush"), { final: true });
    } else if (m.type === "batch") {
      // whole file as fast as possible (no playback): also reports the real-time factor
      call("start", m.sampleRate, m.anc);
      const n = Math.round(m.sampleRate * 0.05);
      const t0 = performance.now();
      for (let i = 0; i < m.samples.length; i += n) {
        reply("result", call("push", m.samples.subarray(i, i + n)));
      }
      const wall = (performance.now() - t0) / 1000;
      reply("result", call("flush"), { final: true, rtf: wall / (m.samples.length / m.sampleRate) });
    }
  } catch (e) {
    postMessage({ type: "error", text: String(e) });
  }
};
