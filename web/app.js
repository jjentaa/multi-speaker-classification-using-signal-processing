// Page logic: audio capture (microphone or file) -> worker running the Python diarizer -> live plots.
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SR = 48000;            // capture rate; the pipeline low-passes and down-samples it to 16 kHz
const FRAME_T = (k) => (k * 160 + 256) / 16000;   // centre time of analysis frame k (10 ms hop, 32 ms window)
const VIEW_S = 30;           // seconds shown while running
const SPEC_PPS = 20;         // spectrogram columns per second
const SPEC_BINS = 170;       // 0-8 kHz at 48 kHz / 1024-point FFT
const SPEC_MAX_S = 1500;
const HUM_HARMONICS = 12;

const worker = new Worker("/web/worker.js");
const S = {
  ready: false, running: false, mode: null,
  sent: 0, total: 0, processed: 0, procMs: null, nSpk: 0, rtf: null,
  energy: [], noise: [], vad: [], f0: [], segments: [], ref: null, refColor: {},
};
let ctx = null, bus = null, capture = null, analyser = null, source = null, micStream = null;
let humNodes = [], specTimer = null, player = null;
const spec = document.createElement("canvas");
spec.width = SPEC_PPS * SPEC_MAX_S;
spec.height = SPEC_BINS;
const specCtx = spec.getContext("2d");
let specUsed = false;

// ---------------------------------------------------------------- worker
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "status") setStatus(m.text);
  else if (m.type === "ready") { S.ready = true; setStatus("Ready", "ok"); syncButtons(); }
  else if (m.type === "error") { setStatus(m.text, "err"); if (S.running) stop(false); }
  else if (m.type === "result") onResult(m);
};

function onResult(m) {
  S.energy.push(...m.energy); S.noise.push(...m.noise); S.vad.push(...m.vad); S.f0.push(...m.f0);
  S.processed = m.time; S.nSpk = m.n_speakers;
  if (!m.final) S.procMs = m.proc_ms;
  m.segments.forEach((seg, i) => { seg.audio = m.audio[i]; S.segments.push(seg); addRow(seg); });
  if (m.segments.length && S.ref) mapReference();
  if (m.final) {
    if (m.rtf != null) S.rtf = m.rtf;
    if (S.mode === "fast") finish();
    setStatus(S.rtf != null ? `Done, ${(1 / S.rtf).toFixed(1)}× faster than real time` : "Done", "ok");
  }
  updateStats();
}

// ---------------------------------------------------------------- audio
async function audioContext() {
  if (!ctx) {
    ctx = new AudioContext({ sampleRate: SR });
    await ctx.audioWorklet.addModule("/web/capture-worklet.js");
  }
  if (ctx.state === "suspended") await ctx.resume();
  if (ctx.sampleRate % 16000) throw new Error(`This browser gave a ${ctx.sampleRate} Hz audio context; a multiple of 16 kHz is needed.`);
  return ctx;
}

function reset(mode) {
  Object.assign(S, { running: true, mode, sent: 0, total: 0, processed: 0, procMs: null, nSpk: 0, rtf: null,
    energy: [], noise: [], vad: [], f0: [], segments: [], refColor: {} });
  $("segments").tBodies[0].replaceChildren();
  $("empty").hidden = false;
  specCtx.clearRect(0, 0, spec.width, spec.height);
  specUsed = false;
  syncButtons();
}

function startGraph(toSpeakers) {
  bus = ctx.createGain();
  capture = new AudioWorkletNode(ctx, "capture", { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1, channelCountMode: "explicit" });
  capture.port.onmessage = (e) => {
    if (!S.running) return;
    S.sent += e.data.length;
    worker.postMessage({ type: "push", chunk: e.data }, [e.data.buffer]);
  };
  analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0;
  bus.connect(capture);
  bus.connect(analyser);
  if ($("hum").checked) {
    const amp = humAmplitude();
    for (let k = 1; k <= HUM_HARMONICS; k++) {
      const osc = ctx.createOscillator(), g = ctx.createGain();
      osc.frequency.value = 50 * k;
      g.gain.value = amp / k;
      osc.connect(g).connect(bus);
      if (toSpeakers) g.connect(ctx.destination);
      osc.start();
      humNodes.push(osc, g);
    }
  }
  const bins = new Uint8Array(analyser.frequencyBinCount);
  const col = specCtx.createImageData(1, SPEC_BINS);
  specTimer = setInterval(() => {
    analyser.getByteFrequencyData(bins);
    for (let b = 0; b < SPEC_BINS; b++) {
      const [r, g, bl] = heat(bins[b] / 255), o = (SPEC_BINS - 1 - b) * 4;
      col.data[o] = r; col.data[o + 1] = g; col.data[o + 2] = bl; col.data[o + 3] = 255;
    }
    const x = Math.floor((S.sent / ctx.sampleRate) * SPEC_PPS);
    if (x < spec.width) { specCtx.putImageData(col, x, 0); specUsed = true; }
  }, 1000 / SPEC_PPS);
}

const humAmplitude = () => (Number($("hum-level").value) / 100) * 0.1;

async function startMic() {
  try {
    await audioContext();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch (e) { setStatus(`Microphone unavailable: ${e.message}`, "err"); return; }
  reset("mic");
  S.ref = null;
  worker.postMessage({ type: "start", sampleRate: ctx.sampleRate, anc: $("anc").checked });
  startGraph(false);
  source = ctx.createMediaStreamSource(micStream);
  source.connect(bus);
  setStatus("Listening", "live");
}

async function playBuffer(buffer, ref) {
  reset("file");
  S.ref = ref;
  S.total = buffer.duration;
  worker.postMessage({ type: "start", sampleRate: ctx.sampleRate, anc: $("anc").checked });
  startGraph(true);
  source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(bus);
  source.connect(ctx.destination);
  source.onended = () => { if (S.running) stop(true); };
  source.start();
  setStatus("Playing", "live");
}

async function loadSample() {
  const name = $("sample").value;
  const [wav, ref] = await Promise.all([
    fetch(`/web/samples/${name}.wav`).then((r) => r.arrayBuffer()),
    fetch(`/web/samples/${name}.json`).then((r) => r.json()),
  ]);
  return { buffer: await ctx.decodeAudioData(wav), ref };
}

async function playSample() {
  try {
    await audioContext();
    setStatus("Loading sample…");
    const { buffer, ref } = await loadSample();
    await playBuffer(buffer, ref);
  } catch (e) { setStatus(String(e.message || e), "err"); }
}

async function runFast() {
  try {
    await audioContext();
    setStatus("Loading sample…");
    const { buffer, ref } = await loadSample();
    const x = new Float32Array(buffer.length);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const d = buffer.getChannelData(c);
      for (let i = 0; i < x.length; i++) x[i] += d[i] / buffer.numberOfChannels;
    }
    if ($("hum").checked) {
      const amp = humAmplitude();
      for (let k = 1; k <= HUM_HARMONICS; k++) {
        const w = (2 * Math.PI * 50 * k) / buffer.sampleRate;
        for (let i = 0; i < x.length; i++) x[i] += (amp / k) * Math.sin(w * i);
      }
    }
    reset("fast");
    S.ref = ref;
    S.total = buffer.duration;
    S.sent = x.length;
    setStatus("Processing…", "live");
    worker.postMessage({ type: "batch", samples: x, sampleRate: buffer.sampleRate, anc: $("anc").checked }, [x.buffer]);
  } catch (e) { setStatus(String(e.message || e), "err"); }
}

async function playUpload(file) {
  try {
    await audioContext();
    setStatus("Decoding…");
    await playBuffer(await ctx.decodeAudioData(await file.arrayBuffer()), null);
  } catch (e) { setStatus(`Could not read that file: ${e.message || e}`, "err"); }
}

function stop(flush = true) {
  if (!S.running) return;
  const wasFast = S.mode === "fast";
  finish();
  if (wasFast) return;                     // the batch job cannot be interrupted; its results keep arriving
  S.total = S.sent / SR;
  clearInterval(specTimer);
  if (source) { source.onended = null; try { source.stop(); } catch (_) { /* a microphone source has no stop() */ } source.disconnect(); }
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  humNodes.forEach((n) => { if (n.stop) n.stop(); n.disconnect(); });
  if (capture) { capture.port.onmessage = null; capture.disconnect(); }
  if (bus) bus.disconnect();
  source = micStream = capture = bus = null;
  humNodes = [];
  if (flush) { worker.postMessage({ type: "flush" }); setStatus("Finishing…"); }
}

function finish() { S.running = false; syncButtons(); }

function playSegment(seg) {
  if (!seg.audio || !seg.audio.length || !ctx) return;
  if (player) { try { player.stop(); } catch (_) { /* already ended */ } }
  const buf = ctx.createBuffer(1, seg.audio.length, 16000);
  buf.copyToChannel(seg.audio, 0);
  player = ctx.createBufferSource();
  player.buffer = buf;
  player.connect(ctx.destination);
  player.start();
}

// ---------------------------------------------------------------- UI
function setStatus(text, cls = "") { const el = $("status"); el.textContent = text; el.className = `pill ${cls}`; }

function syncButtons() {
  const idle = S.ready && !S.running;
  for (const id of ["btn-mic", "btn-sample", "btn-fast", "file", "sample", "anc", "hum"]) $(id).disabled = !idle;
  $("btn-stop").disabled = !S.running || S.mode === "fast";
}

const spkColor = (i) => (i >= 0 ? css(`--spk${(i % 4) + 1}`) : css("--unknown"));

function addRow(seg) {
  $("empty").hidden = true;
  const tr = document.createElement("tr");
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = seg.label;
  chip.style.background = spkColor(seg.speaker);
  const cells = [chip, `${seg.start.toFixed(2)} s`, `${seg.end.toFixed(2)} s`, seg.f0_hz ? `${Math.round(seg.f0_hz)} Hz` : "–",
    seg.reason, `${seg.latency_s.toFixed(2)} s`];
  for (const c of cells) { const td = document.createElement("td"); td.append(c); tr.append(td); }
  const btn = document.createElement("button");
  btn.textContent = "Play";
  btn.onclick = () => playSegment(seg);
  const td = document.createElement("td");
  td.append(btn);
  tr.append(td);
  $("segments").tBodies[0].prepend(tr);
}

function updateStats() {
  const now = currentTime();
  $("s-time").textContent = `${now.toFixed(1)} s`;
  $("s-spk").textContent = S.nSpk;
  $("s-seg").textContent = S.segments.length;
  $("s-lag").textContent = S.mode === "fast" ? "–" : `${Math.max(0, S.sent / SR - S.processed).toFixed(2)} s`;
  $("s-proc").textContent = S.procMs == null ? "–" : `${S.procMs.toFixed(1)} ms`;
}

const currentTime = () => (S.mode === "fast" ? S.processed : S.sent / SR);

// colour each reference speaker like the system speaker it overlaps most (display only)
function mapReference() {
  const pairs = [];
  const refs = [...new Set(S.ref.map((r) => r[2]))];
  for (const r of refs) for (let k = 0; k < 8; k++) {
    let ov = 0;
    for (const [a, b, lab] of S.ref) if (lab === r) for (const s of S.segments) if (s.speaker === k) ov += Math.max(0, Math.min(b, s.end) - Math.max(a, s.start));
    if (ov > 0) pairs.push([ov, r, k]);
  }
  pairs.sort((x, y) => y[0] - x[0]);
  const used = new Set();
  S.refColor = {};
  for (const [, r, k] of pairs) if (!(r in S.refColor) && !used.has(k)) { S.refColor[r] = k; used.add(k); }
}

// ---------------------------------------------------------------- drawing
function heat(v) {
  const stops = [[13, 8, 60], [84, 15, 109], [187, 55, 84], [249, 142, 9], [252, 255, 164]];
  const p = Math.min(0.999, Math.max(0, v)) * (stops.length - 1), i = Math.floor(p), f = p - i;
  return stops[i].map((c, j) => Math.round(c + (stops[i + 1][j] - c) * f));
}

function prep(id) {
  const c = $(id), dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = Number(c.getAttribute("data-h") || c.getAttribute("height"));
  if (!c.hasAttribute("data-h")) c.setAttribute("data-h", h);
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "11px system-ui, sans-serif";
  return { g, w, h };
}

function view() {
  const now = currentTime();
  if (S.running && S.mode !== "fast") { const t1 = Math.max(now, VIEW_S); return [t1 - VIEW_S, t1]; }
  return [0, Math.max(S.total, now, S.processed, 1)];
}

function timeAxis(g, w, h, t0, t1) {
  const span = t1 - t0, step = [1, 2, 5, 10, 20, 30, 60, 120, 300].find((s) => span / s <= 12) || 600;
  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted"); g.lineWidth = 1; g.textAlign = "center";
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    const x = ((t - t0) / span) * w;
    g.beginPath(); g.moveTo(x + 0.5, 0); g.lineTo(x + 0.5, h - 14); g.stroke();
    g.fillText(`${t} s`, Math.min(Math.max(x, 14), w - 14), h - 3);
  }
  g.textAlign = "left";
}

function drawTimeline(t0, t1) {
  const { g, w, h } = prep("c-timeline"), X = (t) => ((t - t0) / (t1 - t0)) * w;
  timeAxis(g, w, h, t0, t1);
  const bar = (a, b, y, color, text) => {
    const x0 = Math.max(0, X(a)), x1 = Math.min(w, X(b));
    if (x1 <= x0) return;
    g.fillStyle = color; g.fillRect(x0, y, Math.max(1, x1 - x0), 24);
    if (text && x1 - x0 > g.measureText(text).width + 8) { g.fillStyle = "#fff"; g.fillText(text, x0 + 4, y + 16); }
  };
  g.fillStyle = css("--muted"); g.fillText("System output", 2, 11);
  for (const s of S.segments) bar(s.start, s.end, 15, spkColor(s.speaker), s.label);
  g.fillStyle = css("--muted");
  g.fillText(S.ref ? "Reference (ground truth, coloured by best-matching system speaker)" : "Reference: not available for this source", 2, 52);
  if (S.ref) for (const [a, b, lab] of S.ref) bar(a, b, 56, lab in S.refColor ? spkColor(S.refColor[lab]) : css("--unknown"), lab);
}

function frameRange(t0, t1, n) {
  return [Math.max(0, Math.floor((t0 * 16000 - 256) / 160)), Math.min(n, Math.ceil((t1 * 16000 - 256) / 160) + 1)];
}

function drawEnergy(t0, t1) {
  const { g, w, h } = prep("c-energy"), X = (t) => ((t - t0) / (t1 - t0)) * w, ph = h - 14;
  const Y = (db) => ph - ((Math.max(-90, Math.min(0, db)) + 90) / 90) * ph;
  const [k0, k1] = frameRange(t0, t1, S.energy.length);
  g.fillStyle = css("--speech");
  for (let k = k0; k < k1; k++) if (S.vad[k]) { let j = k; while (j < k1 && S.vad[j]) j++; g.fillRect(X(FRAME_T(k)), 0, Math.max(1, X(FRAME_T(j)) - X(FRAME_T(k))), ph); k = j; }
  timeAxis(g, w, h, t0, t1);
  const line = (arr, color, dash) => {
    g.strokeStyle = color; g.setLineDash(dash); g.lineWidth = 1.2; g.beginPath();
    for (let k = k0; k < k1; k++) { const x = X(FRAME_T(k)), y = Y(arr[k]); if (k === k0) g.moveTo(x, y); else g.lineTo(x, y); }
    g.stroke(); g.setLineDash([]);
  };
  line(S.energy, css("--ink"), []);
  line(S.noise, css("--spk2"), [4, 3]);
  g.fillStyle = css("--muted"); g.fillText("0 dBFS", 2, 11); g.fillText("−90", 2, ph - 3);
  g.textAlign = "right"; g.fillText("line: energy   dashed: noise floor   shaded: speech", w - 4, 11); g.textAlign = "left";
}

function drawF0(t0, t1) {
  const { g, w, h } = prep("c-f0"), X = (t) => ((t - t0) / (t1 - t0)) * w, ph = h - 14;
  const Y = (hz) => ph - ((hz - 60) / 340) * ph;
  timeAxis(g, w, h, t0, t1);
  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted");
  for (const hz of [100, 200, 300]) { g.beginPath(); g.moveTo(0, Y(hz) + 0.5); g.lineTo(w, Y(hz) + 0.5); g.stroke(); g.fillText(`${hz} Hz`, 2, Y(hz) - 2); }
  const [k0, k1] = frameRange(t0, t1, S.f0.length);
  g.fillStyle = css("--accent");
  for (let k = k0; k < k1; k++) if (S.f0[k] > 0) g.fillRect(X(FRAME_T(k)) - 1, Y(S.f0[k]) - 1, 2, 2);
}

function drawSpec(t0, t1) {
  const { g, w, h } = prep("c-spec"), ph = h - 14;
  if (specUsed) {
    g.imageSmoothingEnabled = true;
    g.drawImage(spec, t0 * SPEC_PPS, 0, Math.max(1, (t1 - t0) * SPEC_PPS), SPEC_BINS, 0, 0, w, ph);
  } else {
    g.fillStyle = css("--muted");
    g.fillText(S.mode === "fast" ? "Not recorded in fast mode (there is no playback to analyse)." : "Appears while audio is running.", 8, 20);
  }
  timeAxis(g, w, h, t0, t1);
}

function frame() {
  const [t0, t1] = view();
  drawTimeline(t0, t1); drawEnergy(t0, t1); drawF0(t0, t1); drawSpec(t0, t1);
  if (S.running) updateStats();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- wiring
$("btn-mic").onclick = startMic;
$("btn-sample").onclick = playSample;
$("btn-fast").onclick = runFast;
$("btn-stop").onclick = () => stop(true);
$("file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) playUpload(f); };
requestAnimationFrame(frame);
