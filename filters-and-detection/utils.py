"""
Shared helpers for the DSP health-signals exercises.

Contains the synthetic signal generators (PPG, ECG, motion-corrupted PPG) and
the reusable filter building blocks (Butterworth band-pass / low-pass), plus a
small helper for saving figures into a common ``figures/`` directory.

Everything here is imported by the four ``exerciseN_*.py`` scripts so the
algorithms live in exactly one place.
"""

import os

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ── Figure output ─────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save_figure(fig, filename, dpi=150):
    """Save ``fig`` as a PNG into the shared ``figures/`` directory.

    Returns the absolute path written, so the caller can print it.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ── Synthetic signal generators ───────────────────────────────
def synth_ppg(fs=100, dur=10.0, hr_bpm=72, seed=0):
    """Synthetic PPG: cardiac pulses + baseline wander (<0.5 Hz) + HF noise (>8 Hz).

    Returns ``(t, signal)``.
    """
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    t = np.arange(N) / fs
    rr = 60.0 / hr_bpm

    clean = np.zeros(N)
    bt = 0.3
    while bt < dur:
        clean += 1.0 * np.exp(-((t - bt) / 0.060) ** 2)
        clean += 0.4 * np.exp(-((t - bt - 0.18) / 0.090) ** 2)
        bt += rr * (1 + 0.03 * rng.standard_normal())

    baseline = 0.6 * np.sin(2 * np.pi * 0.25 * t) \
        + 0.3 * np.sin(2 * np.pi * 0.08 * t)
    hf = 0.15 * np.sin(2 * np.pi * 15 * t) + 0.10 * rng.standard_normal(N)
    return t, clean + baseline + hf


def synth_ecg(fs=360, dur=10.0, hr_bpm=75, seed=1):
    """Synthetic ECG: PQRST beats + baseline wander + 60 Hz powerline + HF noise.

    Returns ``(t, signal, r_peaks)`` where ``r_peaks`` are the ground-truth
    R-wave sample indices (used by the Pan-Tompkins scoring in exercise 2).
    """
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    t = np.arange(N) / fs
    rr_base = 60.0 / hr_bpm

    waves = [(-0.20, 0.10, 0.025),    # P
             (-0.025, -0.15, 0.0125),  # Q
             (0.00, 1.00, 0.010),      # R
             (0.025, -0.25, 0.0125),   # S
             (0.16, 0.35, 0.040)]      # T

    clean = np.zeros(N)
    r_peaks = []
    bt = 0.3
    while bt < dur:
        r_peaks.append(int(round(bt * fs)))   # R wave sits at offset 0.00
        for off, amp, sig in waves:
            clean += amp * np.exp(-((t - bt - off) / sig) ** 2)
        bt += rr_base * (1 + 0.02 * rng.standard_normal())

    baseline = 0.5 * np.sin(2 * np.pi * 0.20 * t) \
        + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    powerline = 0.08 * np.sin(2 * np.pi * 60 * t)
    hf = 0.04 * rng.standard_normal(N)
    return t, clean + baseline + powerline + hf, np.array(r_peaks)


def synth_ppg_motion(fs=64, dur=60.0, seed=0):
    """Synthetic PPG with a rest -> walk -> run transition.

    Three phases:
      0-20 s  : rest  - HR ~1.1 Hz, no motion
      20-40 s : walk  - HR ~1.4 Hz, cadence ~1.8 Hz (steps/s)
      40-60 s : run   - HR ~2.2 Hz, cadence ~2.8 Hz

    Returns ``(t, ppg, acc, card)``:
      ppg  - corrupted PPG (cardiac + motion + noise)
      acc  - accelerometer reference (motion proxy) for the LMS filter
      card - clean cardiac signal (ground truth for comparison)
    """
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    t = np.arange(N) / fs
    ppg = np.zeros(N)
    acc = np.zeros(N)    # accelerometer reference for LMS
    card = np.zeros(N)   # clean cardiac (ground truth)

    for n, tn in enumerate(t):
        if tn < 20:                          # ── REST ──
            hr = 1.1
            motion = 0.0
        elif tn < 40:                        # ── WALK ──
            hr = 1.4
            cadence = 1.8
            env = np.sin(np.pi * (tn - 20) / 20)   # fade in/out
            motion = env * (1.8 * np.sin(2 * np.pi * cadence * tn)
                            + 0.7 * np.sin(2 * np.pi * 2 * cadence * tn))
        else:                                # ── RUN ──
            hr = 2.2
            cadence = 2.8
            env = np.sin(np.pi * (tn - 40) / 20)
            motion = env * (2.5 * np.sin(2 * np.pi * cadence * tn)
                            + 1.0 * np.sin(2 * np.pi * 2 * cadence * tn))

        cardiac = np.sin(2 * np.pi * hr * tn)
        card[n] = cardiac
        acc[n] = motion + 0.05 * rng.standard_normal()
        ppg[n] = cardiac + motion + 0.08 * rng.standard_normal()

    return t, ppg, acc, card


# ── Reusable filters ──────────────────────────────────────────
def bandpass(sig, fs, lo, hi, order=4):
    """Zero-phase Butterworth band-pass (SOS form for numerical stability)."""
    nyq = fs / 2
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def lowpass(sig, fs, cutoff=8.0, order=4):
    """Zero-phase Butterworth low-pass (SOS form)."""
    sos = butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, sig)
