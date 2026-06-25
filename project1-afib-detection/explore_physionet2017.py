"""
Step 3 — the *other* dataset: PhysioNet/CinC Challenge 2017.

afdb teaches us what AFib looks like in long, clean, two-lead recordings. But a
wrist sensor sees something much harder: a *short, single-lead, often noisy*
strip and has to commit to one answer. Challenge 2017 is exactly that setting,
so it's the reality check for Project 1.

What every record is:
  • ONE file pair:  A00001.mat  (the ECG samples) + A00001.hea (the header)
  • single lead, fs = 300 Hz, length VARIES (~9–60 s, most ~30 s)
  • the .mat holds one integer array 'val' — raw ADC units, not millivolts yet

The labels live in ONE file, REFERENCE.csv, with no header — just
``record,label`` per line, where label is a single letter:
  N = Normal      A = AFib      O = Other rhythm      ~ = too Noisy to call

That 4-class setup is the key difference from afdb's per-sample rhythm spans:
here the label is ONE verdict for the WHOLE strip (already "windowed" for us).

What this script does:
  1. Reads REFERENCE.csv and prints the class balance (it's imbalanced).
  2. Dissects ONE record so you can see exactly what's in the .mat and .hea.
  3. Reports the recording-length distribution (why these aren't fixed windows).
  4. Saves follow-along figures: class balance + one example strip per class.

Run:  python explore_physionet2017.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from utils import C2017_DIR, FIGURES_DIR

FOLLOW = os.path.join(FIGURES_DIR, "followalong")
FS = 300                                   # Challenge-2017 sampling rate (Hz)

LABEL_NAMES = {"N": "Normal", "A": "AFib", "O": "Other", "~": "Noisy"}
LABEL_COLORS = {"Normal": "C0", "AFib": "C3", "Other": "C2", "Noisy": "0.5"}


def load_reference():
    """record id → readable label, from REFERENCE.csv (``A00001,N`` per line)."""
    path = os.path.join(C2017_DIR, "REFERENCE.csv")
    if not os.path.exists(path):
        raise SystemExit("No Challenge-2017 data — run: python download_data.py --c2017")
    ref = {}
    for line in open(path):
        rec, code = line.strip().split(",")
        ref[rec] = LABEL_NAMES.get(code, code)
    return ref


def load_ecg(rec):
    """Raw single-lead ECG for one record, as a 1-D float array (ADC units)."""
    mat = loadmat(os.path.join(C2017_DIR, rec + ".mat"))
    return mat["val"].squeeze().astype(float)


def dissect_one(rec, label):
    """Print everything inside one record's two files — 'what is the data?'."""
    print(f"\nDissecting record {rec}  (label: {label})")
    print("  .hea header says:")
    for line in open(os.path.join(C2017_DIR, rec + ".hea")):
        print("      " + line.rstrip())
    sig = load_ecg(rec)
    print(f"  .mat 'val' array: shape {sig.shape}, dtype read as float")
    print(f"      {len(sig)} samples ÷ {FS} Hz = {len(sig)/FS:.1f} s of ECG")
    print(f"      raw range [{sig.min():.0f}, {sig.max():.0f}] ADC units "
          f"(divide by the header's gain for mV)")


def class_balance(ref):
    counts = {}
    for lab in ref.values():
        counts[lab] = counts.get(lab, 0) + 1
    total = len(ref)
    print(f"\nClass balance ({total:,} recordings):")
    for nm in ("Normal", "AFib", "Other", "Noisy"):
        if nm in counts:
            print(f"    {nm:<7} {counts[nm]:>5,}  ({100*counts[nm]/total:5.1f}%)")
    return counts, total


def length_stats(ref):
    """Why these aren't tidy fixed windows: lengths vary a lot."""
    lens = []
    for rec in ref:
        # read length cheaply from the .hea (2nd token on the first line)
        first = open(os.path.join(C2017_DIR, rec + ".hea")).readline().split()
        lens.append(int(first[3]) / FS)
    lens = np.array(lens)
    print(f"\nRecording length: min {lens.min():.0f}s  median {np.median(lens):.0f}s"
          f"  max {lens.max():.0f}s")
    print("  → variable length is why Challenge-2017 is one-label-per-strip,")
    print("    not the fixed 30 s windows we cut afdb into.")
    return lens


def figure_class_balance(counts, total):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [n for n in ("Normal", "AFib", "Other", "Noisy") if n in counts]
    vals = [counts[n] for n in names]
    ax.bar(names, vals, color=[LABEL_COLORS[n] for n in names])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{100*v/total:.0f}%", ha="center", va="bottom")
    ax.set_title(f"Challenge-2017 class balance ({total:,} strips)")
    ax.set_ylabel("# recordings")
    plt.tight_layout()
    os.makedirs(FOLLOW, exist_ok=True)
    path = os.path.join(FOLLOW, "c2017_class_balance.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_example_per_class(ref, seconds=10):
    """One example strip per class, first ~10 s, so each class has a face."""
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for i, nm in enumerate(("Normal", "AFib", "Other", "Noisy")):
        rec = next((r for r, lab in ref.items() if lab == nm), None)
        if rec is None:
            continue
        sig = load_ecg(rec)[: seconds * FS]
        t = np.arange(len(sig)) / FS
        ax[i].plot(t, sig, color=LABEL_COLORS[nm], lw=0.7)
        ax[i].set_title(f"{nm} — record {rec}", loc="left", fontsize=10)
        ax[i].set_ylabel("ADC")
    ax[-1].set_xlabel("Time (s)")
    plt.suptitle("Challenge-2017 — one example strip per class",
                 fontweight="bold")
    plt.tight_layout()
    os.makedirs(FOLLOW, exist_ok=True)
    path = os.path.join(FOLLOW, "c2017_examples.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ref = load_reference()
    counts, total = class_balance(ref)
    # Dissect one AFib record so the example isn't a boring normal strip.
    afib_rec = next(r for r, lab in ref.items() if lab == "AFib")
    dissect_one(afib_rec, "AFib")
    length_stats(ref)
    print(f"\nSaved {figure_class_balance(counts, total)}")
    print(f"Saved {figure_example_per_class(ref)}")


if __name__ == "__main__":
    main()
