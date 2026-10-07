"""Build a test conversation with known speaker turns (ground truth RTTM).

Uses the macOS `say` text-to-speech voices as 1-4 different "speakers",
joins random turns with random pauses (some very short, so the speaker-change
detector has to work without a pause), adds background noise and a little room
reverb, and writes the result at the microphone rate (48 kHz).

    python tools/make_conversation.py --speakers 3 --turns 24 --out data/conv3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarize.io import load_audio  # noqa: E402

VOICES = ["Daniel", "Samantha", "Rishi", "Karen", "Moira", "Tessa", "Aman", "Tara"]
SENTENCES = [
    "I think we should start the meeting with a short review of last week.",
    "The signal is sampled at forty eight kilohertz and then down sampled.",
    "Could you explain why the filter needs to be applied before decimation?",
    "Yes, otherwise the high frequencies fold back and cause aliasing.",
    "That makes sense. What about the window length for each frame?",
    "We use thirty two milliseconds with a hop of ten milliseconds.",
    "I am not sure the pitch estimate is reliable for every speaker.",
    "Autocorrelation works well when the voice is clearly periodic.",
    "Let's test it with four people talking in the same small room.",
    "The results will be saved as a json file with timestamps and labels.",
    "Okay, I will prepare the evaluation script before Friday.",
    "Please remember to measure the delay of the whole system as well.",
    "Right, the target is less than two seconds of latency.",
    "Honestly, the clustering part is the hardest problem here.",
    "We can compare the online labels with the offline result later.",
    "Sounds good to me. Let's move on to the next topic.",
    "Does anyone have questions about the spectrogram plot?",
    "No, but the noise at the beginning looks a bit strange.",
    "That is the fan of my laptop, I will move it further away.",
    "Great, then we are done for today. Thank you everyone.",
]


def tts(text: str, voice: str, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "u.wav")
        subprocess.run(["say", "-v", voice, "-o", p, "--file-format=WAVE", "--data-format=LEI16@22050", text],
                       check=True)
        x, _ = load_audio(p, sr)
    # trim the silence TTS puts around the utterance so the reference is exact
    hop = sr // 100
    e = np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x) - hop, hop)])
    on = np.where(10 * np.log10(e + 1e-12) > 10 * np.log10(e.max()) - 40)[0]
    return x[on[0] * hop:(on[-1] + 1) * hop] if len(on) else x


def room(x: np.ndarray, sr: int, rt60: float, rng) -> np.ndarray:
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0 / 0.25
    ir *= 0.25
    y = np.convolve(x, ir)[: len(x)]
    return y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", type=int, default=3)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--voices", nargs="*", default=None, help="macOS voice names (default: built-in list)")
    ap.add_argument("--snr", type=float, default=30.0, help="background noise SNR in dB")
    ap.add_argument("--rt60", type=float, default=0.2, help="room reverberation time (0 = off)")
    ap.add_argument("--min-pause", type=float, default=0.05)
    ap.add_argument("--max-pause", type=float, default=0.8)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/conv")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    voices = (a.voices or VOICES)[: a.speakers]
    gains = rng.uniform(0.5, 1.0, len(voices))       # people talk at different levels
    parts, turns, t = [np.zeros(int(0.8 * a.sr))], [], 0.8
    prev = -1
    for k in range(a.turns):
        spk = rng.choice([s for s in range(len(voices)) if s != prev]) if len(voices) > 1 else 0
        prev = spk
        text = SENTENCES[rng.integers(len(SENTENCES))]
        if rng.random() < 0.3:  # sometimes a longer turn
            text += " " + SENTENCES[rng.integers(len(SENTENCES))]
        u = tts(text, voices[spk], a.sr) * gains[spk]
        parts.append(u)
        turns.append((t, t + len(u) / a.sr, f"{voices[spk]}"))
        t += len(u) / a.sr
        pause = rng.uniform(a.min_pause, a.max_pause) if rng.random() > 0.3 else rng.uniform(0.0, 0.15)
        parts.append(np.zeros(int(pause * a.sr)))
        t += pause
        print(f"  turn {k + 1:2d}: {voices[spk]:<9} {turns[-1][0]:6.2f}-{turns[-1][1]:6.2f}  pause {pause:.2f}s")
    parts.append(np.zeros(int(1.0 * a.sr)))
    x = np.concatenate(parts)
    if a.rt60 > 0:
        x = room(x, a.sr, a.rt60, rng)
    speech_pow = np.mean(np.concatenate([x[int(s * a.sr):int(e * a.sr)] for s, e, _ in turns]) ** 2)
    x = x + rng.standard_normal(len(x)) * np.sqrt(speech_pow / 10 ** (a.snr / 10))
    x *= 0.5 / np.max(np.abs(x))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + ".wav", x.astype(np.float32), a.sr, subtype="PCM_16")
    uri = os.path.basename(a.out)
    with open(a.out + ".rttm", "w") as f:
        for s, e, lab in turns:
            f.write(f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")
    print(f"wrote {a.out}.wav ({len(x) / a.sr:.1f} s, {len(voices)} speakers) and {a.out}.rttm")


if __name__ == "__main__":
    main()
