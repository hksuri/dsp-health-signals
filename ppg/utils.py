"""
Shared helpers for the PPG exercises.

Holds the synthetic-PPG generator, the PPG-DaLiA dataset loader, and the
figure-saving helper, so the ``exerciseN_*.py`` scripts can stay focused on
the algorithm under study.
"""

import os
import pickle

import numpy as np


# ── PPG-DaLiA dataset ─────────────────────────────────────────
# Activity ID -> name, per the PPG-DaLiA "activity" stream (Reiss et al. 2019).
DALIA_ACTIVITIES = {
    0: "transient",     # between / undefined activity
    1: "sitting",
    2: "stairs",
    3: "table-soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}

# Fixed sampling rates / ground-truth windowing baked into the dataset.
DALIA_FS_BVP = 64        # wrist BVP (PPG), Hz
DALIA_FS_ACT = 4         # activity label stream, Hz
DALIA_HR_WIN_S = 8.0     # ground-truth HR is computed over 8 s windows…
DALIA_HR_SHIFT_S = 2.0   # …sliding by 2 s, so label[k] covers [2k, 2k+8] s


def load_dalia_subject(pkl_path):
    """Load one PPG-DaLiA subject pickle (``S{i}.pkl``).

    The official release stores each subject as a pickled dict::

        d['signal']['wrist']['BVP']  -> (n, 1) wrist PPG @ 64 Hz
        d['label']                   -> (m,)   ECG-derived ground-truth HR (bpm),
                                                one value per 2 s (8 s window)
        d['activity']                -> (k, 1) activity IDs @ 4 Hz

    Returns a dict with flat arrays: ``bvp``, ``hr_true``, ``activity``.
    The pickles were written under Python 2, so ``encoding='latin1'`` is
    required to unpickle them on Python 3.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return {
        "bvp": np.asarray(d["signal"]["wrist"]["BVP"], dtype=float).ravel(),
        "hr_true": np.asarray(d["label"], dtype=float).ravel(),
        "activity": np.asarray(d["activity"]).ravel().astype(int),
        "subject": d.get("subject", os.path.basename(pkl_path)),
    }


def find_dalia_subjects(data_dir):
    """Return sorted ``S{i}.pkl`` paths under a PPG-DaLiA ``PPG_FieldStudy`` dir.

    The archive unzips to ``PPG_FieldStudy/S1/S1.pkl`` … ``S15/S15.pkl``.
    """
    paths = []
    if not os.path.isdir(data_dir):
        return paths
    for name in sorted(os.listdir(data_dir),
                       key=lambda s: (len(s), s)):  # S1, S2, … S10 (not S1, S10, S2)
        pkl = os.path.join(data_dir, name, f"{name}.pkl")
        if os.path.isfile(pkl):
            paths.append(pkl)
    return paths


# ── Figure output ─────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save_figure(fig, filename, dpi=150):
    """Save ``fig`` as a PNG into this folder's ``figures/`` directory.

    Returns the absolute path written, so the caller can print it.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ── Synthetic signal generator ────────────────────────────────
def synth_ppg_quality(fs=64, dur=24.0, hr_bpm=72, seed=0):
    """Synthetic PPG whose signal *quality* changes over time.

    A real optical PPG is a large, slowly-varying **DC** light level (the
    light that makes it back to the photodiode through static tissue) with a
    tiny pulsatile **AC** ripple on top (the extra absorption each heartbeat).
    We model exactly that:

        ppg(t) = DC_baseline(t) + amplitude(t) * pulse(t) + noise(t)

    The pulse amplitude is healthy for the first and last thirds, but collapses
    in the middle third — a "sensor lift-off / poor perfusion" episode where
    almost no pulse reaches the photodiode. That gives us a signal where the
    Perfusion Index *should* drop below a usable threshold, so Exercise 1 can
    demonstrate quality-gating.

    Returns ``(t, ppg)``.
    """
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    t = np.arange(N) / fs
    rr = 60.0 / hr_bpm                      # seconds between beats

    # Per-sample pulse amplitude: healthy ~1.0, but near-zero during 8-13 s.
    amplitude = np.full(N, 1.1)
    bad = (t >= 8.0) & (t < 13.0)
    amplitude[bad] = 0.04

    # Cardiac pulse train: a sharp systolic peak + a smaller dicrotic bump,
    # the same double-Gaussian shape used for the synthetic PPG elsewhere.
    pulse = np.zeros(N)
    bt = 0.3
    while bt < dur:
        pulse += np.exp(-((t - bt) / 0.060) ** 2)          # systolic peak
        pulse += 0.4 * np.exp(-((t - bt - 0.18) / 0.090) ** 2)  # dicrotic notch
        bt += rr * (1 + 0.03 * rng.standard_normal())      # natural HR jitter

    # Large positive DC light level with slow baseline wander (breathing etc.).
    dc_baseline = 100.0 + 3.0 * np.sin(2 * np.pi * 0.08 * t)

    # High-frequency sensor noise, slightly worse during the poor-contact
    # episode (but kept modest — too much in-band noise would itself look like
    # a pulse and partly mask the perfusion drop).
    noise = 0.05 * rng.standard_normal(N)
    noise[bad] += 0.04 * rng.standard_normal(int(bad.sum()))

    ppg = dc_baseline + amplitude * pulse + noise
    return t, ppg


# ── Tachogram sources (for HRV, Exercise 3) ──────────────────
# Standard HRV frequency bands (Task Force 1996), in Hz.
HRV_LF_BAND = (0.04, 0.15)   # low frequency — baroreflex / sympathetic+vagal
HRV_HF_BAND = (0.15, 0.40)   # high frequency — respiratory (vagal) modulation


def synth_tachogram(dur_s=300.0, mean_hr_bpm=60, lf_hz=0.10, hf_hz=0.25,
                    lf_ms=40.0, hf_ms=25.0, jitter_ms=5.0, seed=0):
    """Synthetic NN-interval series (a *tachogram*) with KNOWN LF/HF content.

    A tachogram is the sequence of beat-to-beat intervals (here in ms). Real
    HRV rides two oscillations on top of the mean interval: a slow **LF** wave
    (~0.1 Hz, baroreflex) and a faster **HF** wave (~0.25 Hz, respiratory sinus
    arrhythmia). We inject both at known frequencies and amplitudes so the
    frequency-domain HRV computation can be checked against a ground-truth
    answer — something a real recording can never give you.

    Each interval is evaluated at its own beat time (intervals are sampled
    once per beat, on an *uneven* time grid — the whole point of Exercise 3).

    Returns ``(nn_ms, meta)`` where ``meta`` records the injected parameters.
    """
    rng = np.random.default_rng(seed)
    mean_nn = 60000.0 / mean_hr_bpm        # ms between beats
    nn = []
    t = 0.0                                # beat time, seconds
    while t < dur_s:
        modulation = (lf_ms * np.sin(2 * np.pi * lf_hz * t)
                      + hf_ms * np.sin(2 * np.pi * hf_hz * t))
        interval = mean_nn + modulation + jitter_ms * rng.standard_normal()
        nn.append(interval)
        t += interval / 1000.0             # advance by the interval we just drew
    meta = dict(lf_hz=lf_hz, hf_hz=hf_hz, lf_ms=lf_ms, hf_ms=hf_ms,
                mean_hr_bpm=mean_hr_bpm)
    return np.asarray(nn), meta


# Beat annotation symbols counted as heartbeats (same set as the Pan-Tompkins
# exercise): N,L,R,… are the AAMI beat classes in the MIT-BIH annotations.
_MITBIH_BEAT_LABELS = set("NLRBAaJSVrFejnE/fQ")


def load_mitbih_nn(record="100", pn_dir="mitdb", nn_lo_ms=300.0, nn_hi_ms=2000.0):
    """Real NN intervals from a PhysioNet MIT-BIH record's beat annotations.

    Reads the cardiologist ``.atr`` annotations, keeps the beat symbols, turns
    successive R-peak sample indices into intervals (ms), and drops anything
    outside a physiologic ``[nn_lo_ms, nn_hi_ms]`` window — a crude ectopic /
    artifact filter so a single missed or extra beat can't dominate HRV.

    Requires the ``wfdb`` package and PhysioNet access. Returns ``(nn_ms, fs)``.
    """
    import wfdb
    ann = wfdb.rdann(record, "atr", pn_dir=pn_dir)
    fs = ann.fs
    samp = np.array([s for s, c in zip(ann.sample, ann.symbol)
                     if c in _MITBIH_BEAT_LABELS], dtype=float)
    rr_ms = np.diff(samp) / fs * 1000.0
    nn = rr_ms[(rr_ms >= nn_lo_ms) & (rr_ms <= nn_hi_ms)]
    return nn, fs


# ── Two-wavelength PPG (for SpO₂, Exercise 4) ────────────────
# Standard pulse-oximeter linear calibration: SpO2 ≈ A − B·R, where R is the
# ratio-of-ratios. A=110, B=25 are empirical constants fit to human cohorts —
# not physics. We invert them to *synthesize* data with a known SpO2, then let
# Exercise 4 recover it, which validates the signal processing (the AC/DC ratio
# is self-normalizing), NOT the calibration curve itself.
SPO2_CAL_A = 110.0
SPO2_CAL_B = 25.0


def spo2_to_R(spo2):
    """Invert the linear calibration to get the ratio-of-ratios for a target SpO2."""
    return (SPO2_CAL_A - spo2) / SPO2_CAL_B


def synth_ppg_red_ir(spo2_true, fs=64, dur=20.0, hr_bpm=72,
                     dc_ir=8000.0, dc_red=6000.0, perfusion=0.02,
                     noise_frac=0.002, seed=0):
    """Synthetic red + IR PPG whose ratio-of-ratios encodes ``spo2_true``.

    Each channel is a large **DC** light level with a small pulsatile **AC**
    ripple on top. A pulse oximeter reads oxygen saturation from how the AC/DC
    ratio differs between two wavelengths:

        R = (AC_red / DC_red) / (AC_ir / DC_ir)

    so we build the two channels to hit a chosen ``R = spo2_to_R(spo2_true)``:
    the IR channel gets a perfusion of ``AC_ir/DC_ir = perfusion``, and the red
    channel is scaled so its ratio is ``R × perfusion``. Because saturation
    lives in the *ratio*, the absolute DC and AC levels are free parameters —
    which is exactly the self-normalizing property Exercise 4 demonstrates.

    Returns ``(t, red, ir)``.
    """
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    t = np.arange(N) / fs
    rr = 60.0 / hr_bpm

    # Shared cardiac pulse shape (systolic peak + dicrotic bump), same physiology
    # in both channels — only the amplitude differs by wavelength.
    pulse = np.zeros(N)
    bt = 0.3
    while bt < dur:
        pulse += np.exp(-((t - bt) / 0.060) ** 2)
        pulse += 0.4 * np.exp(-((t - bt - 0.18) / 0.090) ** 2)
        bt += rr * (1 + 0.03 * rng.standard_normal())
    # Normalize so its robust peak-to-peak (p95−p5) is 1 → AC amplitude maps
    # directly onto the scale factors below.
    ptp = np.percentile(pulse, 95) - np.percentile(pulse, 5)
    pulse = (pulse - pulse.mean()) / ptp

    R = spo2_to_R(spo2_true)
    ac_ir = perfusion * dc_ir              # AC_ir / DC_ir = perfusion
    ac_red = R * perfusion * dc_red        # AC_red / DC_red = R × perfusion

    ir = dc_ir + ac_ir * pulse + noise_frac * dc_ir * rng.standard_normal(N)
    red = dc_red + ac_red * pulse + noise_frac * dc_red * rng.standard_normal(N)
    return t, red, ir
