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
AFDB_DB = DB                       # PhysioNet slug, used for remote fallback

# ── Where downloaded data lives (git-ignored; see download_data.py) ──
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
AFDB_DIR = os.path.join(DATA_DIR, "afdb")
C2017_DIR = os.path.join(DATA_DIR, "challenge2017")


def _afdb(rec, ext="hea"):
    """Resolve an afdb record to a (name, kwargs) pair for wfdb.

    Prefer the local download in ``data/afdb`` so scripts run offline and fast;
    fall back to streaming from PhysioNet (``pn_dir=``) if the specific file
    isn't there yet. We check the exact ``ext`` the caller is about to read
    (``hea``/``dat``/``atr``/``qrs``) so a half-finished download — a header
    present but its annotations not yet — falls back to remote instead of
    erroring. Every wfdb reader below funnels through this.
    """
    local = os.path.join(AFDB_DIR, rec)
    if os.path.exists(f"{local}.{ext}"):
        return local, {}
    return rec, {"pn_dir": DB}

# Rhythm aux-note -> readable name. afdb tiles every instant with one of these
# four. Note: "(N" is afdb's catch-all for *all other rhythms*, not strictly
# normal sinus — so "Normal" here means "not AFib/AFlutter/AV-junctional".
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
    """All 25 afdb record names, e.g. '04015'.

    Reads the local ``RECORDS`` manifest if the data is downloaded, else asks
    PhysioNet.
    """
    manifest = os.path.join(AFDB_DIR, "RECORDS")
    if os.path.exists(manifest):
        return [ln.strip() for ln in open(manifest) if ln.strip()]
    return wfdb.get_record_list(DB)


def has_signal(rec):
    """False for the two annotation-only records (00735, 03665)."""
    name, kw = _afdb(rec)
    return wfdb.rdheader(name, **kw).sig_len > 0


def record_length(rec):
    """Record length in samples (falls back to the last beat if no signal)."""
    name, kw = _afdb(rec)
    hdr = wfdb.rdheader(name, **kw)
    if hdr.sig_len > 0:
        return hdr.sig_len
    beats, _ = load_beats(rec)
    return int(beats[-1]) if len(beats) else 0


def load_beats(rec):
    """QRS beat sample indices and fs, from the ``.qrs`` annotations."""
    name, kw = _afdb(rec, "qrs")
    ann = wfdb.rdann(name, "qrs", **kw)
    return np.asarray(ann.sample), ann.fs


def load_signal(rec, sampfrom, sampto, channel=0):
    """One ECG channel over [sampfrom, sampto) — read straight from disk.

    Returns ``(signal, fs, channel_name)``.
    """
    name, kw = _afdb(rec, "dat")
    r = wfdb.rdrecord(name, sampfrom=sampfrom, sampto=sampto,
                      channels=[channel], **kw)
    return r.p_signal[:, 0], r.fs, r.sig_name[0]


# ── Rhythm structure ──────────────────────────────────────────
def rhythm_spans(rec):
    """Rhythm episodes as ``(start, end, name)`` spans covering the record.

    Each ``.atr`` annotation gives a start; the episode runs to the next start
    (or the record end for the last one). Returns ``(spans, fs)``.
    """
    name, kw = _afdb(rec, "atr")
    ann = wfdb.rdann(name, "atr", **kw)
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


# ── Windowing: from expert episodes → fixed-length labelled windows ──
# The dataset labels rhythm as variable-length *episodes* (see rhythm_spans).
# A classifier, though, wants fixed-length *windows*. This is the bridge:
# tile the record into ``win_s``-second windows and give each a single label
# derived from whatever episode(s) it overlaps.
WINDOW_DTYPE = [("start", "i8"), ("end", "i8"), ("label", "U16"),
                ("afib_frac", "f4"), ("pure", "?")]


def window_labels(rec, win_s=30.0, hop_s=None, rule="majority"):
    """Tile a record into fixed windows, each carrying one rhythm label.

    Returns ``(windows, fs)`` where ``windows`` is a structured array with:
      start, end   — window bounds in samples
      label        — the propagated rhythm (see ``rule``)
      afib_frac    — fraction of the window's labelled time that is AFib
      pure         — True if the window sits inside a single episode

    ``rule`` decides how to collapse a window that straddles >1 episode:
      "majority"     — the rhythm covering the most of the window (default)
      "afib_if_any"  — AFib if *any* part is AFib (high-sensitivity labelling)
      "pure"         — keep the rhythm only if the window is 100% one episode,
                       otherwise label it "Mixed" (drop these to train clean)

    Windows are non-overlapping by default (``hop_s = win_s``).
    """
    spans, fs = rhythm_spans(rec)
    if not spans:
        return np.array([], dtype=WINDOW_DTYPE), fs

    win = int(win_s * fs)
    hop = int((hop_s or win_s) * fs)
    rec_start, rec_end = spans[0][0], spans[-1][1]

    rows = []
    w = rec_start
    while w + win <= rec_end:
        we = w + win
        cover = {}                                   # rhythm name → samples
        for s, e, nm in spans:
            overlap = min(e, we) - max(s, w)
            if overlap > 0:
                cover[nm] = cover.get(nm, 0) + overlap
        total = sum(cover.values())
        afib_frac = cover.get("AFib", 0) / total if total else 0.0
        majority = max(cover, key=cover.get)
        pure = len(cover) == 1

        if rule == "majority":
            label = majority
        elif rule == "afib_if_any":
            label = "AFib" if afib_frac > 0 else majority
        elif rule == "pure":
            label = majority if pure else "Mixed"
        else:
            raise ValueError(f"unknown rule {rule!r}")

        rows.append((w, we, label, afib_frac, pure))
        w += hop

    return np.array(rows, dtype=WINDOW_DTYPE), fs
