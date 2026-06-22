"""
Helpers for exploring the MIT-BIH Atrial Fibrillation Database (``afdb``).

Everything here is a thin, readable wrapper around ``wfdb`` so the exploration
scripts stay short. Signals are read in *slices* straight from PhysioNet
(partial remote reads), so nothing downloads a full 10-hour record.

afdb facts worth knowing:
  • 25 records, two ECG channels, fs = 250 Hz, ~10 h each.
  • ``.atr`` holds rhythm episodes — each annotation marks where a rhythm starts
    and runs until the next one. Aux notes: (N, (AFIB, (AFL, (J.
  • ``.qrs`` holds the (machine-detected) beat locations — our source of RR.
  • Records 00735 and 03665 have annotations but NO signal file.
"""

import os

import numpy as np
import wfdb

DB = "afdb"

# Rhythm aux-note -> readable name (the four rhythms annotated in afdb).
RHYTHM_NAMES = {
    "(N": "Normal",
    "(AFIB": "AFib",
    "(AFL": "Aflutter",
    "(J": "AV junctional",
}

# Consistent colors so every figure reads the same way.
RHYTHM_COLORS = {"Normal": "C0", "AFib": "C3", "Aflutter": "C1",
                 "AV junctional": "C2"}


# ── Figure output ─────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save_figure(fig, filename, dpi=150):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ── Loaders ───────────────────────────────────────────────────
def list_records():
    """All 25 afdb record names, e.g. '04015'."""
    return wfdb.get_record_list(DB)


def has_signal(rec):
    """False for the two annotation-only records (00735, 03665)."""
    return wfdb.rdheader(rec, pn_dir=DB).sig_len > 0


def record_length(rec):
    """Record length in samples (falls back to the last beat if no signal)."""
    hdr = wfdb.rdheader(rec, pn_dir=DB)
    if hdr.sig_len > 0:
        return hdr.sig_len
    beats, _ = load_beats(rec)
    return int(beats[-1]) if len(beats) else 0


def load_beats(rec):
    """QRS beat sample indices and fs, from the ``.qrs`` annotations."""
    ann = wfdb.rdann(rec, "qrs", pn_dir=DB)
    return np.asarray(ann.sample), ann.fs


def load_signal(rec, sampfrom, sampto, channel=0):
    """One ECG channel over [sampfrom, sampto) — a partial remote read.

    Returns ``(signal, fs, channel_name)``.
    """
    r = wfdb.rdrecord(rec, pn_dir=DB, sampfrom=sampfrom, sampto=sampto,
                      channels=[channel])
    return r.p_signal[:, 0], r.fs, r.sig_name[0]


# ── Rhythm structure ──────────────────────────────────────────
def rhythm_spans(rec):
    """Rhythm episodes as ``(start, end, name)`` spans covering the record.

    Each ``.atr`` annotation gives a start; the episode runs to the next start
    (or the record end for the last one). Returns ``(spans, fs)``.
    """
    ann = wfdb.rdann(rec, "atr", pn_dir=DB)
    starts = [int(s) for s in ann.sample]
    names = [RHYTHM_NAMES.get(a.strip().strip("\x00"), a.strip())
             for a in ann.aux_note]
    end = record_length(rec)
    spans = []
    for i, (s, name) in enumerate(zip(starts, names)):
        e = starts[i + 1] if i + 1 < len(starts) else end
        if e > s:
            spans.append((s, e, name))
    return spans, ann.fs


def beats_with_rhythm(rec):
    """Beats tagged by the rhythm they fall in.

    Returns ``(beats, fs, labels)`` where ``labels[i]`` is the rhythm name at
    ``beats[i]`` (or None if it falls outside any annotated span).
    """
    beats, fs = load_beats(rec)
    spans, _ = rhythm_spans(rec)
    labels = np.full(len(beats), None, dtype=object)
    for s, e, name in spans:
        labels[(beats >= s) & (beats < e)] = name
    return beats, fs, labels


def rr_intervals(beats, fs, lo_s=0.3, hi_s=2.0):
    """RR intervals (s) between successive beats, kept to a physiologic range.

    Returns ``(rr, keep)`` where ``keep`` is a boolean mask over ``diff(beats)``
    so the caller can align per-interval rhythm labels.
    """
    rr = np.diff(beats) / fs
    keep = (rr >= lo_s) & (rr <= hi_s)
    return rr, keep


def find_episode(spans, name, min_seconds, fs):
    """First rhythm span of ``name`` lasting at least ``min_seconds``.

    Returns ``(start, end)`` samples, or None if there isn't one.
    """
    need = int(min_seconds * fs)
    for s, e, nm in spans:
        if nm == name and (e - s) >= need:
            return s, e
    return None
