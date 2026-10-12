"""Checks that each DSP block does what the theory says."""
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarize import dsp  # noqa: E402
from diarize.config import Config  # noqa: E402
from diarize.gauss import GaussStats, delta_bic  # noqa: E402
from diarize.pipeline import StreamingDiarizer  # noqa: E402

rng = np.random.default_rng(0)


def test_dft_matches_fft():
    x = rng.standard_normal(64)
    assert np.allclose(dsp.dft(x), np.fft.fft(x))


def test_hamming_matches_scipy():
    assert np.allclose(dsp.hamming(512), signal.get_window("hamming", 512, fftbins=False))


def test_autocorrelation_is_linear_not_circular():
    x = rng.standard_normal((1, 100))
    direct = np.correlate(x[0], x[0], "full")[99:]
    assert np.allclose(dsp.autocorrelation(x)[0], direct)


def test_streaming_decimator_equals_batch():
    x = rng.standard_normal(48000)
    batch = dsp.Decimator(48000, 16000).process(x)
    d = dsp.Decimator(48000, 16000)
    cuts = [0, 1234, 1235, 20001, 33333, 48000]
    stream = np.concatenate([d.process(x[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    assert np.allclose(batch, stream)
    assert len(batch) == 16000


def test_decimator_removes_aliasing():
    fs, t = 48000, np.arange(48000) / 48000
    tone_in_band = np.sin(2 * np.pi * 1000 * t)
    tone_alias = np.sin(2 * np.pi * 12000 * t)   # would fold to 4 kHz without the low-pass
    y = dsp.Decimator(fs, 16000).process(tone_in_band + tone_alias)[1000:]
    P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    p1k = P[np.argmin(abs(f - 1000))]
    p4k = P[np.argmin(abs(f - 4000))]
    assert 10 * np.log10(p1k / p4k) > 50


def test_streaming_filters_equal_batch():
    x = rng.standard_normal(5000)
    for mk in (lambda: dsp.PreEmphasis(0.97), lambda: dsp.Butterworth(4, (80, 7000), 16000, "bandpass")):
        whole = mk().process(x)
        f = mk()
        parts = np.concatenate([f.process(x[:1700]), f.process(x[1700:1701]), f.process(x[1701:])])
        assert np.allclose(whole, parts)


def test_preemphasis_is_highpass():
    w, h = signal.freqz([1, -0.97], 1, worN=[0, np.pi])
    assert abs(h[0]) < 0.05 and abs(h[1]) > 1.9


def test_pitch_of_pulse_train():
    fs = 16000
    for f0 in (90.0, 150.0, 240.0):
        n = np.arange(512)
        # glottal-like pulse train filtered by a formant resonance
        x = (np.mod(n, fs / f0) < 1).astype(float)
        x = signal.lfilter([1], [1, -1.3, 0.8], x)
        est, v = dsp.pitch_autocorr(x[None, :], fs)
        assert abs(est[0] - f0) / f0 < 0.03, (f0, est)
        assert v[0] > 0.35


def test_noise_is_unvoiced():
    est, v = dsp.pitch_autocorr(rng.standard_normal((20, 512)), 16000)
    assert np.mean(est == 0) > 0.8


def test_bic_same_vs_different():
    a = rng.standard_normal((200, 4))
    b = rng.standard_normal((200, 4))
    c = rng.standard_normal((200, 4)) * 2 + 3
    assert delta_bic(GaussStats.from_data(a), GaussStats.from_data(b)) < 0
    assert delta_bic(GaussStats.from_data(a), GaussStats.from_data(c)) > 0


def test_pipeline_chunk_size_invariant():
    """Output must not depend on how the audio is chopped into blocks."""
    fs = 48000
    t = np.arange(3 * fs) / fs
    x = 0.3 * signal.sawtooth(2 * np.pi * 120 * t) * (np.sin(2 * np.pi * 3 * t) > -0.2)
    x = np.concatenate([np.zeros(fs // 2), x, np.zeros(fs)]) + 1e-3 * rng.standard_normal(4 * fs + fs // 2)
    res = []
    for chunk in (480, 2400, 7777):
        d = StreamingDiarizer(Config(), input_sr=fs)
        for i in range(0, len(x), chunk):
            d.push(x[i:i + chunk])
        d.flush()
        res.append([(round(s.start, 3), round(s.end, 3), s.speaker) for s in d.segments])
    assert res[0] and res[0] == res[1] == res[2]


def test_adaptive_canceller_removes_hum_keeps_noise():
    fs = 16000
    t = np.arange(6 * fs) / fs
    hum = 0.2 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t + 1.0))
    noise = 0.01 * rng.standard_normal(len(t))
    e = dsp.AdaptiveNoiseCanceller(taps=512, delay=480, mu=0.02, fs=fs).process(hum + noise)
    tail = slice(4 * fs, None)
    assert np.mean((e - noise)[tail] ** 2) < 0.01 * np.mean(hum[tail] ** 2)   # hum down > 20 dB
    assert np.mean(e[tail] ** 2) > 0.5 * np.mean(noise[tail] ** 2)            # unpredictable part kept


def test_adaptive_canceller_chunk_size_invariant():
    x = np.sin(2 * np.pi * 50 * np.arange(8000) / 16000) + 0.1 * rng.standard_normal(8000)
    a = dsp.AdaptiveNoiseCanceller(taps=64, delay=160)
    b = dsp.AdaptiveNoiseCanceller(taps=64, delay=160)
    whole = a.process(x)
    parts = np.concatenate([b.process(x[i:i + 777]) for i in range(0, len(x), 777)])
    assert np.allclose(whole, parts)


def test_kaiser_lowpass_meets_its_specification():
    fs, fp, fstop, delta = 48000, 6800.0, 8000.0, 0.001
    h = dsp.kaiser_lowpass(fp, fstop, delta, fs)
    assert len(h) % 2 == 1 and np.allclose(h, h[::-1])              # type I linear phase
    f, H = signal.freqz(h, worN=8192, fs=fs)
    assert np.max(np.abs(np.abs(H[f <= fp]) - 1)) < 1.5 * delta     # passband ripple
    assert np.max(np.abs(H[f >= fstop])) < 1.5 * delta              # stopband attenuation (60 dB)


def test_median_filter_removes_spikes_and_is_chunk_invariant():
    x = np.full(50, 120.0)
    x[[10, 30]] = 240.0                                             # isolated octave errors
    y = dsp.MedianFilter(5).process(x)
    assert np.all(y[4:] == 120.0)
    m = dsp.MedianFilter(5)
    assert np.allclose(np.concatenate([m.process(x[i:i + 7]) for i in range(0, 50, 7)]), y)


def test_normal_equations_recover_ar_coefficients():
    from diarize.ar import itakura, prediction_error_filter
    a = np.array([1.2, -0.6, 0.1])                                  # x[n] = 1.2x[n-1] - 0.6x[n-2] + 0.1x[n-3] + w[n]
    x = signal.lfilter([1.0], np.concatenate([[1.0], -a]), rng.standard_normal(200000))
    r = dsp.autocorr_lags(x[None, :], 3)[0]
    A = prediction_error_filter(r / r[0], reg=0.0)
    assert np.allclose(-A[1:], a, atol=0.02)
    assert abs(itakura(r / r[0], r / r[0])) < 1e-9                  # same spectrum -> distance 0
    y = signal.lfilter([1.0], [1.0, -0.5], rng.standard_normal(200000))
    ry = dsp.autocorr_lags(y[None, :], 3)[0]
    assert itakura(r / r[0], ry / ry[0]) > 0.1                      # different spectrum -> clearly positive


def test_omp_recovers_a_sparse_code():
    from diarize.sparse import omp
    D = rng.standard_normal((12, 20))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    A = np.zeros((30, 12))
    for row in A:
        row[rng.choice(12, 2, replace=False)] = rng.uniform(1, 2, 2) * rng.choice([-1, 1], 2)
    X = A @ D
    A_hat, R = omp(D, X, 2)
    assert np.allclose(A_hat, A, atol=1e-6) and np.max(np.abs(R)) < 1e-6


def test_ksvd_dictionary_fits_its_own_data_better_than_other_data():
    from diarize.sparse import ksvd, residual_ratio
    def cloud(seed):                                   # frames scattered along 4 directions of a 16-d space
        r = np.random.default_rng(seed)
        base = r.standard_normal((4, 16))
        return base[r.integers(0, 4, 400)] * r.uniform(0.5, 2, (400, 1)) + 0.05 * r.standard_normal((400, 16))
    a, b = cloud(1), cloud(2)
    D = ksvd(a[:300], 4, 1)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    assert residual_ratio(D, a[300:], 1) < 0.05        # unseen frames of the same source
    assert residual_ratio(D, b, 1) > 5 * residual_ratio(D, a[300:], 1)
