# Real-time Multi-speaker Diarization

ระบบแยกและระบุผู้พูด (1–4 คน ไม่พูดซ้อนกัน) แบบ real-time จากไมโครโฟนคอมพิวเตอร์ โดยใช้ signal processing ล้วนๆ
(numpy / scipy) ไม่มี neural network เลย ทุกครั้งที่ผู้พูดเปลี่ยน ระบบจะตัดเสียงของคนก่อนหน้าออกมาเป็นไฟล์ `.wav`
ที่ทำความสะอาดแล้ว (พร้อมส่งต่อให้ ASR) และบันทึก timestamp + label ลง JSON

## ติดตั้ง

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

(หรือ `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`)

## วิธีใช้

**Real-time จากไมโครโฟน**  (กด Ctrl+C เพื่อหยุด)

```bash
.venv/bin/python main.py live --out runs/live
```

บน macOS ครั้งแรกต้องอนุญาตให้ Terminal ใช้ไมโครโฟน (System Settings → Privacy & Security → Microphone)
ดูรายชื่ออุปกรณ์ได้ด้วย `main.py devices` แล้วเลือกด้วย `--device <index>`

**จากไฟล์เสียง** (ป้อนเข้าไปทีละ 50 ms เหมือนไมค์จริง ใช้โค้ดชุดเดียวกัน) พร้อมประเมินผลและพล็อตกราฟ

```bash
.venv/bin/python main.py file data/test4.wav --ref data/test4.rttm --out runs/test4 --plot --refine
```

**ประเมินผลลัพธ์ที่มีอยู่แล้ว** (เช่นจากการอัดสด ที่ทำ reference เองทีหลัง)

```bash
.venv/bin/python main.py eval --ref my_reference.rttm --hyp runs/live/result.json
```

**สร้างบทสนทนาทดสอบที่รู้คำตอบ** (ใช้เสียง TTS ของ macOS เป็นผู้พูด, มี noise + reverb)

```bash
.venv/bin/python tools/make_conversation.py --speakers 3 --turns 20 --out data/conv3
```

**จูนพารามิเตอร์** (grid search หา DER ต่ำสุด) และ **unit test**

```bash
.venv/bin/python tools/tune.py data/conv2 data/conv3 data/conv4 --grid bic_lambda=1.5,1.7,1.9 clu_lambda=1.4,1.8,2.2
```

```bash
.venv/bin/python -m pytest tests
```

พารามิเตอร์ทุกตัวอยู่ใน [diarize/config.py](diarize/config.py) และ override จาก command line ได้ เช่น
`--set min_silence_s=0.4 bic_win_s=1.2`

### Output

```
runs/<name>/
  result.json          # timestamp + label ของทุก segment (ข้อ 2.2)
  result.rttm          # รูปแบบมาตรฐานสำหรับ pyannote.metrics
  segments/0003_SPK_2_00012.40-00015.93.wav   # 1 ไฟล์ = 1 ผู้พูด, 16 kHz, ทำความสะอาดแล้ว
  eval.json, session.png, session_timeline.png   # เมื่อใส่ --ref / --plot
```

```json
{"id": 0, "speaker": 0, "label": "SPK_1", "start": 0.821, "end": 3.631, "duration": 2.81,
 "confidence": 1.0, "f0_hz": 189.4, "reason": "change", "emitted_at": 4.9, "latency_s": 1.278,
 "file": "segments/0000_SPK_1_00000.82-00003.63.wav"}
```

`reason` บอกว่าตัดเพราะอะไร: `pause` (เงียบ ≥ 0.3 s), `change` (ΔBIC ตรวจเจอการเปลี่ยนผู้พูดกลางเสียงพูด),
`max` (ยาวเกิน 20 s), `flush` (หยุดอัด)

## Pipeline (ตรงกับหัวข้อ 2.1 ใน proposal)

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

| หัวข้อ | ทำอะไร | โค้ด |
|---|---|---|
| 2.1.1 Sampling | รับเสียงจากไมค์ 48 kHz ทีละบล็อก 50 ms ผ่าน `sounddevice` callback | [main.py](main.py) `cmd_live` |
| 2.1.2 Low-pass | FIR 97 taps (Hamming window design) ตัดที่ 7.2 kHz กัน aliasing ก่อนลด rate | [dsp.py](diarize/dsp.py) `Decimator` |
| 2.1.3 Down sampling | เก็บทุกตัวอย่างที่ 3 → 16 kHz (จำ phase ข้ามบล็อก จึงได้ผลเท่ากับทำทั้งไฟล์) | `Decimator.process` |
| 2.1.4 Pre-emphasis | `y[n] = x[n] − 0.97 x[n−1]` ยกความถี่สูงให้ formant ชัด | `PreEmphasis` |
| 2.1.5 IIR Butterworth | อันดับ 4 แบบ SOS มี state ต่อเนื่อง ใช้ 3 ตัว: band speech, band pitch, high-pass สำหรับไฟล์เสียง | `Butterworth` |
| 2.1.6 Windowing | Hamming 32 ms, hop 10 ms | `hamming`, `frame_signal` |
| 2.1.7 DFT | ใช้ FFT 512 จุด (มี `dft()` แบบนิยามตรง O(N²) ไว้ตรวจว่าตรงกัน) | `dft`, `power_spectrum` |
| 2.1.8 Power spectrum + Spectrogram | `|X[k]|²/N` → mel filterbank 26 ช่อง → log → DCT → MFCC c1–c12 | `power_spectrum`, `spectrogram`, `mfcc_from_power` |
| 2.1.9 Autocorrelation | Wiener–Khinchin `r = IDFT{|X|²}` (zero-pad ≥ 2N−1 ให้เป็น linear correlation) + center clipping → F0 | `autocorrelation`, `pitch_autocorr` |
| 2.1.10 Clustering | Gaussian ของ MFCC + log-F0 ต่อผู้พูด ตัดสินด้วย ΔBIC (online) และ agglomerative BIC (offline เปรียบเทียบ) | [cluster.py](diarize/cluster.py), [gauss.py](diarize/gauss.py) |

### การสร้าง window ตัดเสียงแบบ real-time ([pipeline.py](diarize/pipeline.py))

state machine ทำงานทุก 10 ms:

1. **VAD** – พลังงานเฟรมเทียบกับ noise floor ที่ปรับตัวเอง (ลงเร็ว ขึ้นช้า) ถ้าเกิน 10 dB = เสียงพูด
2. **ตัดที่ช่วงเงียบ** – เงียบ ≥ 0.3 s → ปิด segment ส่งออกทันที (delay ≈ 0.35 s)
3. **ตัดกลางเสียงพูด** – ผู้พูดบางคนต่อกันโดยไม่เว้นช่วง จึงเทียบหน้าต่าง 1 s สองอันติดกันด้วย
   ΔBIC = ½[N log|Σ| − N₁ log|Σ₁| − N₂ log|Σ₂|] − λP ถ้า ΔBIC > 0 แล้วผ่านจุดสูงสุดไป 0.25 s
   → ตัดที่จุดสูงสุด (delay ≈ 1.3 s)
4. segment ที่ปิดแล้วเข้า **clustering** → ได้ label → **cleaner** (spectral subtraction จาก noise print
   ช่วงเงียบล่าสุด, ตัดหัวท้าย, normalize −20 dBFS, fade) → เขียนไฟล์ + JSON ทันที

### Clustering

แต่ละผู้พูดเป็น full-covariance Gaussian ของ MFCC + Gaussian 1 มิติของ log-F0 (สะสม sufficient statistics
จึงอัปเดต/รวมได้เร็ว) segment ใหม่เข้าผู้พูดที่ ΔBIC ต่ำสุดถ้า ΔBIC < 0 ไม่อย่างนั้นเปิดผู้พูดใหม่ (สูงสุด 4)
โดยกำหนดว่าต้องมีเสียงพูด ≥ 1.5 s ถึงจะเปิดผู้พูดใหม่ได้ (covariance จาก segment สั้นๆ ไม่น่าเชื่อถือ
ทำให้เกิดผู้พูดปลอม) และถ้าโมเดลผู้พูด 2 คนกลายเป็นคล้ายกัน (ΔBIC < 0) จะรวมเป็นคนเดียว

## ผลการทดลอง (ข้อ 4)

บทสนทนาสังเคราะห์จากเสียง TTS ของ macOS (~100 s, 20–24 turns, 30% ของการสลับผู้พูดเว้นช่วง < 0.15 s,
มี noise + room reverb) — ชุด `conv*` ใช้จูนพารามิเตอร์, ชุด `test*` ใช้เสียงอีกชุดและไม่ได้ใช้จูน
(DER ใช้ collar 0.25 s, F1 = macro F1 ต่อผู้พูดหลัง optimal mapping, frame 10 ms)

| ไฟล์ | ผู้พูด | SNR / RT60 | DER | Speaker F1 | ผู้พูดที่เจอ | delay เฉลี่ย / สูงสุด |
|---|---|---|---|---|---|---|
| conv2 | 2 | 30 dB / 0.2 s | 1.8 % | 97.9 % | 2 | 0.93 / 1.32 s |
| conv3 | 3 | 30 dB / 0.2 s | 3.8 % | 94.7 % | 3 | 1.07 / 1.32 s |
| conv4 | 4 | 30 dB / 0.2 s | 3.6 % | 95.5 % | 4 | 1.03 / 1.32 s |
| **test2** | 2 | 30 dB / 0.2 s | **5.8 %** | **93.5 %** | 2 | 0.97 / 1.32 s |
| **test3** | 3 | 25 dB / 0.2 s | **5.9 %** | **93.7 %** | 3 | 0.85 / 1.31 s |
| **test4** | 4 | 20 dB / 0.3 s | **3.9 %** | **96.4 %** | 4 | 0.67 / 1.32 s |

- **Time delay** ไม่เกิน 1.32 s ทุกไฟล์ (เป้าหมาย 1–2 s) — ตัดที่ช่วงเงียบ ~0.35 s, ตัดด้วย ΔBIC ~1.3 s
- ประมวลผลเร็วกว่า real-time ~65 เท่า (real-time factor 0.015 บน MacBook)
- ระบบ online ได้ผลใกล้เคียง/ดีกว่า offline re-clustering (`--refine`) เล็กน้อย

![session](docs/session_test4.png)

## ข้อจำกัด / สิ่งที่ควรทำต่อ

- ตัวเลขข้างบนมาจากเสียง TTS ซึ่งแต่ละ "คน" เสียงคงที่มาก กับคนจริงในห้องควรอัดเสียงจริง (`main.py live`
  จะเก็บ `recording.wav` ไว้) ทำ reference RTTM แล้วใช้ `tools/tune.py` จูน `bic_lambda`, `clu_lambda`,
  `min_silence_s` ใหม่ — ค่า λ ของ BIC มีผลมากที่สุด
- ช่วงแรกของการประชุม ผู้พูดที่พูดสั้นกว่า 1.5 s จะยังไม่ถูกแยกเป็นคนใหม่ (ถูกนับรวมกับคนที่ใกล้ที่สุด)
- label ที่ส่งออกไปแล้วแก้ไม่ได้ (เป็น real-time) ถ้าโมเดลสองคนถูกรวมกันทีหลัง label ใหม่จะถูกต้อง แต่ของเก่าไม่เปลี่ยน
- ตามขอบเขต 1.3.2 ระบบไม่รองรับการพูดซ้อนกัน
