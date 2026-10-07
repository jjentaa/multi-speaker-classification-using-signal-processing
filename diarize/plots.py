"""Figures for the report: waveform + VAD, spectrogram, pitch, dBIC and timelines."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from . import dsp  # noqa: E402
from .config import Config  # noqa: E402

COLORS = ["#2a78d6", "#1baa7a", "#eda20c", "#d44c7a", "#7d5bd3", "#6b7280"]


def _timeline(ax, turns, title):
    labels = sorted({t[2] for t in turns})
    for s, e, l in turns:
        k = labels.index(l)
        ax.barh(k, e - s, left=s, height=0.7, color=COLORS[k % len(COLORS)])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title, loc="left", fontsize=10)
    ax.invert_yaxis()


def plot_session(x: np.ndarray, fs_in: int, trace: dict, cfg: Config, hyp, ref=None, path="session.png"):
    fs = cfg.target_sr
    x16 = signal.resample_poly(x, 1, fs_in // fs) if fs_in != fs else x
    n = len(trace["energy_db"])
    tf = (np.arange(n) * cfg.hop_len + cfg.win_len / 2) / fs
    rows = 7 if ref else 6
    fig, axs = plt.subplots(rows, 1, figsize=(14, 1.9 * rows + 1), sharex=True)

    ax = axs[0]
    ax.plot(np.arange(len(x16)) / fs, x16, lw=0.4, color="#555")
    v = np.asarray(trace["vad"], dtype=float)
    ax.fill_between(tf, -1, 1, where=v > 0, color=COLORS[0], alpha=0.15, step="mid", label="VAD speech")
    ax.set_ylim(-np.max(np.abs(x16)) * 1.05, np.max(np.abs(x16)) * 1.05)
    ax.set_title("Waveform (16 kHz after low-pass + down-sampling) and VAD", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1]
    ax.plot(tf, trace["energy_db"], lw=0.7, label="frame energy")
    ax.plot(tf, trace["noise_db"], lw=1.0, label="noise floor")
    ax.plot(tf, np.asarray(trace["noise_db"]) + cfg.vad_margin_db, lw=0.8, ls="--", label="threshold")
    ax.set_ylabel("dBFS")
    ax.set_title("Energy VAD", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axs[2]
    t, f, S = dsp.spectrogram(x16, fs, cfg.win_len, cfg.hop_len, cfg.n_fft)
    ax.pcolormesh(t, f / 1000, S, shading="auto", cmap="magma", vmin=S.max() - 80, vmax=S.max())
    ax.set_ylabel("kHz")
    ax.set_title("Power spectrogram (Hamming, 512-pt DFT)", loc="left", fontsize=10)

    ax = axs[3]
    f0 = np.asarray(trace["f0"])
    ax.scatter(tf[f0 > 0], f0[f0 > 0], s=1.5, color=COLORS[3])
    ax.set_ylabel("Hz")
    ax.set_ylim(0, cfg.f0_max * 1.05)
    ax.set_title("F0 from autocorrelation", loc="left", fontsize=10)

    ax = axs[4]
    ax.plot(trace["bic_t"], trace["bic"], lw=0.8, color=COLORS[4])
    ax.axhline(0, color="k", lw=0.5)
    for s in hyp:
        if s.reason == "change":
            ax.axvline(s.end, color=COLORS[3], lw=1)
    ax.set_title("dBIC between adjacent 1 s windows (red = detected speaker change)", loc="left", fontsize=10)
    ax.set_ylim(np.nanpercentile(trace["bic"], 1) if trace["bic"] else -1, (np.nanmax(trace["bic"]) + 10) if trace["bic"] else 1)

    hyp_turns = [(s.start, s.end, s.label) for s in hyp]
    if ref:
        _timeline(axs[5], ref, "Reference")
    _timeline(axs[-1], hyp_turns, "System output")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    if ref:
        fig, axs = plt.subplots(2, 1, figsize=(14, 3.6), sharex=True)
        _timeline(axs[0], ref, "Reference")
        _timeline(axs[1], hyp_turns, "System output (online, real-time)")
        axs[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(path.replace(".png", "_timeline.png"), dpi=130)
        plt.close(fig)
