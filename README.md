# Real-time Multi-speaker Diarization

Real-time speaker separation and identification (1–4 speakers, no overlapping speech) from a computer microphone,
using pure signal processing (numpy / scipy) with no neural network at all. Every time the speaker changes, the system
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

**Tune parameters** (grid search for the lowest DER) and **unit tests**

```bash
.venv/bin/python tools/tune.py data/conv2 data/conv3 data/conv4 --grid bic_lambda=1.5,1.7,1.9 clu_lambda=1.4,1.8,2.2
```

```bash
.venv/bin/python -m pytest tests
```

All parameters live in [diarize/config.py](diarize/config.py) and can be overridden from the command line, e.g.
`--set min_silence_s=0.4 bic_win_s=1.2`

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

`reason` says why the segment was cut: `pause` (silence ≥ 0.3 s), `change` (ΔBIC detected a speaker change mid-speech),
`max` (longer than 20 s), `flush` (recording stopped)

## Pipeline (matches section 2.1 of the proposal)

```
mic 48 kHz ─► FIR low-pass 7.2 kHz ─► ↓3 ─► 16 kHz ─┬─► pre-emphasis ─► Butterworth BP 80–7000 ─► Hamming ─► DFT ─► |X|² ─► mel ─► MFCC ─┐
 (2.1.1)        (2.1.2)              (2.1.3)        │     (2.1.4)          (2.1.5)                (2.1.6)  (2.1.7) (2.1.8)             │
                                                    ├─► Butterworth BP 60–900 ─► Hamming ─► autocorrelation ─► F0 ────────────────────────┤
                                                    │                                          (2.1.9)                                  │
                                                    └─► Butterworth HP 70 ─► energy VAD ─► real-time window ◄── ΔBIC change detection ◄─┤
                                                                                                │                                       │
                                                             clean (spectral subtraction) ◄─ cut segment ─► online BIC clustering ◄─────┘
                                                                     │                                          (2.1.10)
                                                                     ▼
                                                             segments/*.wav + result.json
```

| Section | What it does | Code |
|---|---|---|
| 2.1.1 Sampling | Captures 48 kHz microphone audio in 50 ms blocks through a `sounddevice` callback | [main.py](main.py) `cmd_live` |
| 2.1.2 Low-pass | 97-tap FIR (Hamming window design) with a 7.2 kHz cutoff to prevent aliasing before the rate reduction | [dsp.py](diarize/dsp.py) `Decimator` |
| 2.1.3 Down sampling | Keeps every 3rd sample → 16 kHz (the phase is carried across blocks, so the result equals processing the whole file) | `Decimator.process` |
| 2.1.4 Pre-emphasis | `y[n] = x[n] − 0.97 x[n−1]` boosts high frequencies so the formants stand out | `PreEmphasis` |
| 2.1.5 IIR Butterworth | 4th order in SOS form with continuous state; three are used: speech band, pitch band, and a high-pass for audio files | `Butterworth` |
| 2.1.6 Windowing | Hamming 32 ms, hop 10 ms | `hamming`, `frame_signal` |
| 2.1.7 DFT | 512-point FFT (a direct-definition O(N²) `dft()` is kept to verify they match) | `dft`, `power_spectrum` |
| 2.1.8 Power spectrum + Spectrogram | `|X[k]|²/N` → 26-channel mel filterbank → log → DCT → MFCC c1–c12 | `power_spectrum`, `spectrogram`, `mfcc_from_power` |
| 2.1.9 Autocorrelation | Wiener–Khinchin `r = IDFT{|X|²}` (zero-padded to ≥ 2N−1 so it is a linear correlation) + center clipping → F0 | `autocorrelation`, `pitch_autocorr` |
| 2.1.10 Clustering | Per-speaker Gaussian of MFCC + log-F0, decided by ΔBIC (online) and agglomerative BIC (offline, for comparison) | [cluster.py](diarize/cluster.py), [gauss.py](diarize/gauss.py) |

### Building the real-time cutting window ([pipeline.py](diarize/pipeline.py))

A state machine runs every 10 ms:

1. **VAD** – frame energy against a self-adapting noise floor (falls fast, rises slowly); more than 10 dB above it = speech
2. **Cut at a pause** – silence ≥ 0.3 s → close the segment and emit it immediately (delay ≈ 0.35 s)
3. **Cut mid-speech** – some speakers follow each other with no gap, so two adjacent 1 s windows are compared with
   ΔBIC = ½[N log|Σ| − N₁ log|Σ₁| − N₂ log|Σ₂|] − λP; if ΔBIC > 0 and the peak is 0.25 s behind
   → cut at the peak (delay ≈ 1.3 s)
4. A closed segment goes to **clustering** → gets a label → **cleaner** (spectral subtraction using a noise print
   from the most recent silence, trim head and tail, normalize to −20 dBFS, fade) → file + JSON written immediately

### Clustering

Each speaker is a full-covariance Gaussian of MFCC + a 1-D Gaussian of log-F0 (sufficient statistics are accumulated,
so updating/merging is fast). A new segment joins the speaker with the lowest ΔBIC if ΔBIC < 0; otherwise a new speaker
is opened (at most 4), provided there is ≥ 1.5 s of speech (a covariance from a short segment is unreliable and
creates spurious speakers). If two speaker models become similar (ΔBIC < 0) they are merged into one.

## Experimental results (item 4)

Conversations synthesized from macOS TTS voices (~100 s, 20–24 turns, 30% of speaker changes have a gap < 0.15 s,
with noise + room reverb) — the `conv*` set was used for parameter tuning; the `test*` set uses different voices and was not used for tuning
(DER uses a 0.25 s collar, F1 = per-speaker macro F1 after optimal mapping, 10 ms frames)

| File | Speakers | SNR / RT60 | DER | Speaker F1 | Speakers found | Mean / max delay |
|---|---|---|---|---|---|---|
| conv2 | 2 | 30 dB / 0.2 s | 1.8 % | 97.9 % | 2 | 0.93 / 1.32 s |
| conv3 | 3 | 30 dB / 0.2 s | 3.8 % | 94.7 % | 3 | 1.07 / 1.32 s |
| conv4 | 4 | 30 dB / 0.2 s | 3.6 % | 95.5 % | 4 | 1.03 / 1.32 s |
| **test2** | 2 | 30 dB / 0.2 s | **5.8 %** | **93.5 %** | 2 | 0.97 / 1.32 s |
| **test3** | 3 | 25 dB / 0.2 s | **5.9 %** | **93.7 %** | 3 | 0.85 / 1.31 s |
| **test4** | 4 | 20 dB / 0.3 s | **3.9 %** | **96.4 %** | 4 | 0.67 / 1.32 s |

- **Time delay** is at most 1.32 s on every file (target 1–2 s) — pause cuts ~0.35 s, ΔBIC cuts ~1.3 s
- Processing runs ~65× faster than real time (real-time factor 0.015 on a MacBook)
- The online system performs on par with / slightly better than offline re-clustering (`--refine`)

![session](docs/session_test4.png)

## Limitations / future work

- The numbers above come from TTS voices, where each "person" has a very stable voice. For real people in a room, record
  real audio (`main.py live` keeps a `recording.wav`), make a reference RTTM, then use `tools/tune.py` to re-tune
  `bic_lambda`, `clu_lambda`, `min_silence_s` — the BIC λ has the largest effect
- Early in a meeting, a speaker who talks for less than 1.5 s is not yet split off as a new speaker (they are assigned to the nearest one)
- Labels that have already been emitted cannot be changed (it is real-time); if two models are merged later, new labels are correct but old ones stay as they were
- Per scope 1.3.2, the system does not support overlapping speech
