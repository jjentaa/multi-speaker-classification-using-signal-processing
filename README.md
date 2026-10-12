# Real-time Multi-speaker Diarization

Real-time speaker separation and identification (1–4 speakers, no overlapping speech) from a computer microphone,
using pure signal processing (numpy / scipy) with no neural network at all: FIR / IIR filters, an NLMS adaptive filter,
autocorrelation pitch and linear-prediction (AR) speaker models. Every time the speaker changes, the system
cuts the previous speaker's audio out as a cleaned `.wav` file (ready to hand to an ASR) and logs the timestamp + label to JSON.

## Installation

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

(or `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`)

## Usage

**Real-time from the microphone**  (press Ctrl+C to stop)

```bash
.venv/bin/python main.py live --out runs/live
```

On macOS the first run requires granting Terminal microphone access (System Settings → Privacy & Security → Microphone).
List the available devices with `main.py devices` and pick one with `--device <index>`.

**From an audio file** (fed in 50 ms at a time like a real microphone, using the same code path), with evaluation and plots

```bash
.venv/bin/python main.py file data/test4.wav --ref data/test4.rttm --out runs/test4 --plot --refine
```

**Evaluate an existing result** (e.g. from a live recording whose reference was made afterwards)

```bash
.venv/bin/python main.py eval --ref my_reference.rttm --hyp runs/live/result.json
```

**Generate a test conversation with known ground truth** (macOS TTS voices as speakers, with noise + reverb)

```bash
.venv/bin/python tools/make_conversation.py --speakers 3 --turns 20 --out data/conv3
```

**Tune parameters** (grid search for the lowest DER), **hum robustness test** and **unit tests**

```bash
.venv/bin/python tools/tune.py data/conv2 data/conv3 data/conv4 --grid ar_change_thr=0.9,1.1,1.3 ar_clu_thr=0.18,0.22,0.26
```

```bash
.venv/bin/python tools/hum_test.py data/test2 data/test3 data/test4 --snr 20 10 0
```

```bash
.venv/bin/python -m pytest tests
```

All parameters live in [diarize/config.py](diarize/config.py) and can be overridden from the command line, e.g.
`--set min_silence_s=0.4 ar_change_thr=1.2`

**Web demo** (the same Python package running in the browser through Pyodide; microphone, sample conversations or an uploaded file)

```bash
python3 web/serve.py
```

then open `http://localhost:8765/web/`. The site is static, so it can be deployed as is (e.g. `vercel` from the
repository root; `vercel.json` maps `/` to `web/index.html`).

### Output

```
runs/<name>/
  result.json          # timestamp + label of every segment (item 2.2)
  result.rttm          # standard format for pyannote.metrics
  segments/0003_SPK_2_00012.40-00015.93.wav   # 1 file = 1 speaker, 16 kHz, cleaned
  eval.json, session.png, session_timeline.png   # when --ref / --plot is given
```

```json
{"id": 0, "speaker": 0, "label": "SPK_1", "start": 0.821, "end": 3.631, "duration": 2.81,
 "confidence": 1.0, "f0_hz": 189.4, "reason": "change", "emitted_at": 4.9, "latency_s": 1.278,
 "file": "segments/0000_SPK_1_00000.82-00003.63.wav"}
```

`reason` says why the segment was cut: `pause` (silence ≥ 0.3 s), `change` (the prediction error jumped mid-speech: speaker change),
`max` (longer than 20 s), `flush` (recording stopped)

## Pipeline

Every signal-processing block comes from the course slides (chapters 4, 5 and 6); the reference is given in the table.

```
mic 48 kHz ─► Kaiser FIR low-pass ─► ↓3 ─► 16 kHz ─► NLMS canceller ─┬─► pre-emphasis ─► Hamming ─► autocorrelation ─► AR model ─► spectral envelope ─┐
                                                                     ├─► Butterworth BP 60–900 ─► centre clipping ─► autocorrelation ─► F0 ───────────┤
                                                                     └─► Butterworth HP 70 ─► energy VAD ─► median filter ─► real-time window ◄───────┤
                                                                                                                 │ (pause, or change of speaker)      │
                                                                                 clean (Wiener filter) ◄─ cut segment ─► dictionary per speaker ◄─────┘
                                                                                         │                               (K-SVD; OMP, lowest residual wins)
                                                                                         ▼
                                                                                 segments/*.wav + result.json
```

| Block | What it does | Course slides | Code |
|---|---|---|---|
| Sampling | Captures 48 kHz microphone audio in 50 ms blocks through a `sounddevice` callback | – | [main.py](main.py) `cmd_live` |
| FIR low-pass | Kaiser-window design from a specification: passband 6.8 kHz, stopband 8 kHz, δ = 0.001 → A = 60 dB, β = 5.65, M = 146 (147 taps, type I linear phase) | CH04-03 Kaiser window method | [dsp.py](diarize/dsp.py) `kaiser_lowpass`, `Decimator` |
| Down-sampling | Keeps every 3rd sample → 16 kHz (the phase is carried across blocks, so the result equals processing the whole file) | CH04-01 basic structures | `Decimator.process` |
| NLMS canceller | Adaptive FIR (1024 taps, reference = the microphone delayed 30 ms) that removes mains hum / fan tones | CH05-02 adaptive filtering, CH05-03 LMS | `AdaptiveNoiseCanceller` |
| Pre-emphasis | `y[n] = x[n] − 0.97 x[n−1]`, a 2-tap FIR with one zero | CH04-01 FIR vs. IIR | `PreEmphasis` |
| Windowing | Hamming, 32 ms frames, 10 ms hop | CH04-03 commonly used windows | `hamming`, `frame_signal` |
| AR model | Autocorrelation r[0..16] of each frame, then the normal equations R a = r give the linear-prediction coefficients; the frame is described by the log spectral envelope −log\|A(e^jω)\|² at 32 frequencies | CH05-01 AR model / linear prediction, CH05-02 normal equation | `autocorr_lags`, [ar.py](diarize/ar.py), [sparse.py](diarize/sparse.py) `envelope` |
| Butterworth IIR | 4th order, cascade of second-order sections with continuous state: band-pass 60–900 Hz for pitch, high-pass 70 Hz for the output audio | CH04-02 Butterworth, bilinear transformation; CH04-01 cascade form | `Butterworth` |
| Centre clipping + autocorrelation | A clipper removes the formant structure, the autocorrelation peak gives F0 | CH04-03 clippers | `center_clip`, `pitch_autocorr` |
| Energy VAD | Frame energy against a self-adapting noise floor | – | [vad.py](diarize/vad.py) |
| Median filter | Running median on two decision tracks: loud/quiet (5 frames) and "which known speaker fits the last second" (9 checks) | CH04-03 median filters | `MedianFilter`, [vad.py](diarize/vad.py), [pipeline.py](diarize/pipeline.py) `_id_check` |
| Speaker identification | A dictionary of 16 envelope atoms per speaker, learned with K-SVD; a segment's frames are sparse-coded with OMP and go to the speaker whose dictionary leaves the smallest residual | CH06-01 sparse coding, OMP, K-SVD | [sparse.py](diarize/sparse.py), [cluster.py](diarize/cluster.py) |
| Cleaner | Wiener gain S / (S + N) per frequency, noise N from the most recent silence | CH05-02 Wiener filter, signal denoising | [cleaner.py](diarize/cleaner.py) |

### Speaker model: sparse coding with a learned dictionary ([sparse.py](diarize/sparse.py))

Each frame is first turned into the log spectral envelope of its AR(16) model, sampled at 32 frequencies with the
mean removed. A speaker is then a dictionary D of 16 envelope shapes ("atoms"), learned from that speaker's own
frames with **K-SVD**: alternate (1) sparse coding of every frame with **OMP** and (2) updating one atom at a time,
with its coefficients, as the best rank-1 fit (SVD) of the error left for the frames that use it. A frame x is
coded as x ≈ Dᵀα with at most one non-zero coefficient, and the share of energy that cannot be represented,
‖x − Dᵀα‖² / ‖x‖², says how well the frame fits that speaker. A segment is coded with every speaker's dictionary;
the average residual share, plus 0.5 × the difference of mean log-F0, is the distance.

One AR model per speaker ([ar.py](diarize/ar.py), `--set speaker_model=ar`) is a single average envelope; a
dictionary keeps several (vowels, consonants). With the correct turn boundaries, 1 s pieces of speech are
attributed to the right speaker 95.5 % of the time with the dictionary against 91.1 % with one AR model.

### Building the real-time cutting window ([pipeline.py](diarize/pipeline.py))

A state machine runs every 10 ms:

1. **VAD** – frame energy against a self-adapting noise floor (falls fast, rises slowly); more than 10 dB above it
   = loud, and a 5-frame median filter turns that into the speech decision
2. **Cut at a pause** – silence ≥ 0.3 s → close the segment and emit it immediately (delay ≈ 0.35 s)
3. **Cut mid-speech**, checked every 50 ms, because some speakers follow each other with no gap:
   - *between known speakers*: the last 1 s is coded with every speaker's dictionary and the winner is noted; that
     track flickers, so it is median filtered over 9 checks, and the segment is cut where the filtered winner
     differs from the speaker the segment started with (delay ≈ 0.8 s)
   - *to an unknown speaker*: a small dictionary (8 atoms) is learned on each of two adjacent 1 s windows; if each
     codes the other window clearly worse than its own (plus the jump of mean log-F0) → cut at the peak (delay ≈ 1.3 s)
4. A closed segment goes to **clustering** → gets a label → **cleaner** (Wiener filter using a noise print
   from the most recent silence, trim head and tail, normalize to −20 dBFS, fade) → file + JSON written immediately

### Adaptive noise canceller ([dsp.py](diarize/dsp.py) `AdaptiveNoiseCanceller`)

Same structure as an acoustic echo canceller, but a single microphone has no separate reference signal, so the
reference is the microphone itself delayed by D = 30 ms (an adaptive line enhancer):

```
d(n) = v(n) + y(n)        microphone = speech + periodic interference (hum, fan)
x(n) = d(n − D)           reference
ŷ(n) = ĥᵀ x(n)            adaptive FIR estimate of the interference
e(n) = d(n) − ŷ(n)        output, goes on to the three feature branches
ĥ ← ĥ + μ e(n) x(n) / (‖x(n)‖² + L e(n)²)        NLMS, μ = 0.005, L = 1024
```

A periodic interference can be predicted from 30 ms ago, so it is subtracted. Voiced speech and reverberation are
also partly predictable over 30 ms, so the weights are only updated while the output level is within 6 dB of its
noise floor (i.e. nobody is talking), the same reason an echo canceller freezes during double-talk. Set
`--set anc_taps=0` to switch the stage off.

### Clustering

A new segment joins the speaker with the smallest distance if it is below 0.21; otherwise a new speaker is opened
(at most 4), provided there is ≥ 1.5 s of speech. Two speaker models closer than 0.17 (each coded with the other's
dictionary) are merged into one. A speaker keeps its most recent 2000 frames (20 s) and its dictionary is re-learned
when new frames arrive.

### Earlier model (MFCC + Gaussian + ΔBIC)

The first version of this project used MFCC features, a Gaussian per speaker compared with ΔBIC, a Hamming-window
FIR and spectral subtraction, none of which are in the course slides. That code is still in the package
([gauss.py](diarize/gauss.py)) and can be run for comparison with
`--set speaker_model=gauss aa_window=hamming vad_median=0 clean_method=subtract`.

## Experimental results (item 4)

Conversations synthesized from macOS TTS voices (~100 s, 20–24 turns, 30% of speaker changes have a gap < 0.15 s,
with noise + room reverb) — the `conv*` set was used for parameter tuning; the `test*` set uses different voices and was not used for tuning
(DER uses a 0.25 s collar, F1 = per-speaker macro F1 after optimal mapping, 10 ms frames)

| File | Speakers | SNR / RT60 | DER | Speaker F1 | Speakers found | Mean / max delay | DER, one AR model | DER, earlier model |
|---|---|---|---|---|---|---|---|---|
| conv2 | 2 | 30 dB / 0.2 s | 0.7 % | 98.6 % | 2 | 0.78 / 1.36 s | 3.7 % | 1.8 % |
| conv3 | 3 | 30 dB / 0.2 s | 5.7 % | 92.4 % | 4 | 0.76 / 1.36 s | 12.2 % | 3.8 % |
| conv4 | 4 | 30 dB / 0.2 s | 16.2 % | 81.9 % | 4 | 0.72 / 1.35 s | 25.3 % | 3.6 % |
| **test2** | 2 | 30 dB / 0.2 s | **18.6 %** | **85.7 %** | 4 | 0.77 / 1.35 s | 12.1 % | 5.8 % |
| **test3** | 3 | 25 dB / 0.2 s | **4.5 %** | **95.7 %** | 4 | 0.70 / 1.36 s | 25.2 % | 5.8 % |
| **test4** | 4 | 20 dB / 0.3 s | **27.6 %** | **60.8 %** | 4 | 0.84 / 1.39 s | 25.2 % | 4.6 % |

- Mean DER is 7.5 % on `conv*` and 16.9 % on `test*`; almost all of it is speaker confusion (missed speech ≤ 2.4 %,
  false alarm ≤ 0.1 %). One AR model per speaker gave 13.7 % / 20.8 %, and the earlier MFCC + Gaussian model
  (not course content) 3.1 % / 5.4 %.
- The system still opens too many speakers (4 found on four of the six files) and is sensitive to its thresholds:
  a small change can move the DER of one file by 10–20 points, so single-file numbers should not be read finely.
- **Time delay** is at most 1.39 s on every file (target 1–2 s)
- Processing runs ~8× faster than real time (real-time factor 0.13 on a MacBook); learning dictionaries costs about
  twice the time of the single-AR-model version
- Two things were tried and left out because they did not help: dropping a new speaker that nothing matches again
  (CALLHOME DER 44.0 → 44.6 %), and a 5-frame median filter on the F0 track (`f0_median`, off)

**Real telephone conversations** (CALLHOME English, first 10 conversations of the Hugging Face `talkbank/callhome`
set, 76 min, 2–3 speakers, about 9 % of speech overlapped; nothing was tuned on it), mean DER:

| Dictionary (current) | One AR model | Earlier model | Everything labelled as one speaker |
|---|---|---|---|
| 35.1 % | 42.7 % | 29.8 % | 33.1 % |

The dictionary model is 7.6 points better than one AR model here, but still not better than calling everything one
speaker: it beats that baseline on 2 of the 10 conversations, and on 3 it found only one speaker.

The hum and AMI tables below were measured with the single-AR-model version and have not been re-run.

**Mains hum** (50 Hz + harmonics to 600 Hz added to the same six files, `tools/hum_test.py`), mean DER:

| Speech-to-hum ratio | Canceller off | Canceller on |
|---|---|---|
| no hum | 23.3 % | 17.3 % |
| 20 dB | 22.1 % | 20.3 % |
| 10 dB | 32.3 % | 18.7 % |
| 0 dB | 98.4 % | 22.8 % |

With the canceller on, the DER stays near its no-hum level down to a 0 dB speech-to-hum ratio; without it the system
fails completely at 0 dB. The differences in the first two rows are within the run-to-run sensitivity of the speaker
model (a single file changes by 10–20 points when the input is perturbed slightly).

The figure below was made with the earlier MFCC + Gaussian model.

![session](docs/session_test4.png)

**Real meetings** (AMI corpus, headset mix, 4 speakers, 14–25 min, 5–16 % of speech overlapped; default parameters):

| Meeting | DER | miss / false alarm / confusion | Speaker F1 | DER if everything is labelled as one speaker |
|---|---|---|---|---|
| ES2004a | 73.9 % | 26.3 / 8.3 / 39.4 | 22.2 % | 56.2 % |
| IS1009a | 71.2 % | 18.4 / 7.8 / 45.0 | 34.9 % | 38.6 % |
| TS3003a | 69.6 % | 40.3 / 1.1 / 28.3 | 19.4 % | 14.1 % |

On real meetings the system is worse than labelling all speech as one speaker (the last column assumes perfect
speech detection). Turns are short (median ≈ 1 s, below the 1.5 s needed to open a new speaker) and quiet speakers
are missed by the energy VAD. The earlier MFCC + Gaussian model was also below that baseline (74 / 48 / 61 %).

## Limitations / future work

- The numbers above come from TTS voices, where each "person" has a very stable voice. For real people in a room, record
  real audio (`main.py live` keeps a `recording.wav`), make a reference RTTM, then use `tools/tune.py` to re-tune
  `ar_change_thr`, `ar_change_pitch`, `ar_clu_thr`, `min_silence_s` — the change threshold has the largest effect
- The AR speaker model is sensitive: small changes of a threshold or of the input can move the DER of one file by
  10–20 points, and it finds an extra speaker on some files
- Early in a meeting, a speaker who talks for less than 1.5 s is not yet split off as a new speaker (they are assigned to the nearest one)
- Labels that have already been emitted cannot be changed (it is real-time); if two models are merged later, new labels are correct but old ones stay as they were
- The adaptive canceller only removes steady periodic interference (hum, fan tones) and has only been tested on
  synthetic hum; it does nothing for broadband noise, and after a sudden change in the interference it needs a
  pause in the speech to re-converge
- Per scope 1.3.2, the system does not support overlapping speech
