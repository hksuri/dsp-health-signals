"""
Shared helpers for the PPG exercises.

Holds the synthetic-PPG generator and the figure-saving helper, so the
``exerciseN_*.py`` scripts can stay focused on the algorithm under study.
"""

import os

import numpy as np


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
