"""Parameters for the whole pipeline (one place to tune everything)."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- 2.1.1 Sampling / 2.1.2 Low-pass / 2.1.3 Down-sampling -----------------
    input_sr: int = 48000          # microphone sampling rate
    target_sr: int = 16000         # rate used for all analysis (speech band <= 8 kHz)
    aa_window: str = "kaiser"      # FIR design: "kaiser" (from the specification below) or "hamming" (fixed length)
    aa_passband_hz: float = 6800.0 # Kaiser: passband edge
    aa_stopband_hz: float = 8000.0 # Kaiser: stopband edge (= new Nyquist frequency)
    aa_ripple: float = 0.001       # Kaiser: allowed error delta in both bands (60 dB)
    aa_numtaps: int = 97           # Hamming: FIR length
    aa_cutoff_ratio: float = 0.9   # Hamming: cutoff = ratio * target_sr/2  (7.2 kHz)

    # --- Adaptive filtering (NLMS noise canceller, runs on the 16 kHz signal) ---
    anc_taps: int = 1024           # adaptive FIR length (64 ms); 0 = stage off
    anc_delay_s: float = 0.03      # reference = microphone delayed by this much
    anc_mu: float = 0.005          # NLMS step size
    anc_gate_db: float = 6.0       # adapt only while output < noise floor + gate

    # --- 2.1.4 Pre-emphasis -----------------------------------------------------
    preemph: float = 0.97          # y[n] = x[n] - a*x[n-1]

    # --- 2.1.5 IIR Butterworth --------------------------------------------------
    bp_order: int = 4
    speech_band: tuple = (80.0, 7000.0)   # feature branch (after pre-emphasis)
    pitch_band: tuple = (60.0, 900.0)     # autocorrelation / F0 branch
    audio_hp: float = 70.0                # output-audio branch (DC / rumble removal)

    # --- 2.1.6 Windowing / 2.1.7 DFT / 2.1.8 Power spectrum ---------------------
    win_len: int = 512             # 32 ms Hamming window
    hop_len: int = 160             # 10 ms hop  -> 100 frames / s
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 13               # c0..c12, c0 is dropped -> 12 features
    ar_order: int = 16             # order p of the AR (linear prediction) model
    fmin: float = 60.0
    fmax: float = 7600.0

    # --- 2.1.9 Autocorrelation pitch -------------------------------------------
    f0_min: float = 60.0
    f0_max: float = 400.0
    clip_ratio: float = 0.3        # centre clipping level (fraction of peak)
    voicing_thr: float = 0.35      # normalised autocorrelation peak for "voiced"
    f0_median: int = 0             # median filter length on the F0 track (frames, 0 = off)

    # --- Voice activity detection ----------------------------------------------
    vad_margin_db: float = 10.0    # speech if energy > noise floor + margin
    vad_abs_floor_db: float = -55.0
    vad_onset_frames: int = 3      # 30 ms of energy before "speech" (used when vad_median = 0)
    vad_median: int = 5            # median filter length on the loud/quiet decision (frames, 0 = off)
    noise_rise_rate: float = 0.0007 # how fast the noise floor may rise (per frame)
    noise_fall_rate: float = 0.2

    # --- Real-time window (segment cutting) -------------------------------------
    min_silence_s: float = 0.30    # pause that closes a segment
    max_segment_s: float = 20.0    # force a cut (keeps ASR chunks <= 20 s)
    pad_s: float = 0.08            # context kept around every cut
    min_segment_s: float = 0.25    # shorter speech bursts are discarded

    # speaker-change detection inside continuous speech (Delta-BIC)
    bic_win_s: float = 1.0         # analysis window on each side of the split
    bic_step_frames: int = 5       # evaluate every 50 ms
    bic_lambda: float = 1.7        # penalty weight
    bic_patience_s: float = 0.25   # wait this long after the peak before cutting
    bic_min_speech: int = 50       # speech frames required on each side

    # --- Speaker model ----------------------------------------------------------
    speaker_model: str = "dict"    # "dict": learned dictionary + sparse coding, "ar": one AR model, "gauss": MFCC Gaussian + dBIC
    dict_atoms: int = 16           # dict: atoms per speaker (K-SVD)
    dict_sparsity: int = 1         # dict: atoms used per frame (OMP)
    dict_nfreq: int = 32           # dict: points of the log spectral envelope that is coded
    dict_thr: float = 0.21         # dict: residual share above which a segment is a different speaker
    dict_merge_thr: float = 0.17   # dict: two speaker models closer than this are merged
    dict_pitch_weight: float = 0.5 # dict: weight of |difference of mean log-F0|
    dict_change_atoms: int = 8     # dict: atoms of the small dictionaries used for change detection
    dict_change_thr: float = 0.36  # dict: change score between adjacent windows that marks a speaker change
    dict_change_pitch: float = 1.0 # dict: weight of the mean log-F0 jump in the change score
    ar_clu_thr: float = 0.22       # AR: distance above which a segment is a different speaker
    ar_change_thr: float = 1.1     # AR: prediction-error jump between adjacent windows that marks a speaker change
    ar_change_pitch: float = 4.0   # AR: weight of the mean log-F0 jump in the change score
    id_median: int = 9             # median (majority) filter length on the per-check speaker decision, 0 = off
    ar_merge_thr: float = 0.10     # AR: two speaker models closer than this are merged into one
    ar_pitch_weight: float = 1.0   # AR: weight of |difference of mean log-F0|

    # --- 2.1.10 Clustering ------------------------------------------------------
    max_speakers: int = 4
    clu_lambda: float = 1.8        # BIC penalty weight for speaker assignment
    pitch_weight: float = 1.0      # weight of the log-F0 dBIC term (autocorrelation pitch)
    clu_min_new_s: float = 1.5     # speech needed before a *new* speaker may be created
    clu_max_frames: int = 3000     # speaker model memory (older stats decay)

    # --- Output cleaning (for ASR) ----------------------------------------------
    clean_method: str = "wiener"   # "wiener": gain S/(S+N), "subtract": spectral subtraction
    clean_oversub: float = 2.0     # spectral subtraction over-subtraction factor
    clean_floor: float = 0.05      # spectral floor
    clean_target_dbfs: float = -20.0
    noise_memory_s: float = 2.0    # amount of recent non-speech audio kept as noise print
    ring_seconds: float = 60.0     # audio history kept for cutting segments

    @property
    def feature_dims(self) -> int:
        if self.speaker_model == "gauss":
            return self.n_mfcc - 1
        return self.ar_order + (self.dict_nfreq if self.speaker_model == "dict" else 0)

    @property
    def frame_rate(self) -> float:
        return self.target_sr / self.hop_len
