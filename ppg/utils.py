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
