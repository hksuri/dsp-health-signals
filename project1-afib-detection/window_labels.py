"""
Step 2 — turn expert rhythm *episodes* into fixed-length labelled *windows*.

A detector trains on fixed-length windows, but afdb labels rhythm as
variable-length episodes — so somewhere we have to window the signal and
propagate a label to each window. It's worth understanding exactly what
happens here, because it's where the dataset's shape changes:

  BEFORE (how afdb stores truth):  variable-length episodes
      |————— Normal —————|—— AFib ——|———— Normal ————|     (minutes to hours)

  AFTER (what a classifier eats):  fixed 30 s windows, one label each
      [N][N][N][N][?][AF][AF][?][N][N][N] …

The only hard part is the boundary windows ([?] above) — a 30 s window that
lands half on Normal and half on AFib. How we label those is a *design choice*,
not a fact in the data, so this script makes that choice visible and lets you
compare the options before committing.

What it does:
  1. Windows ONE record and prints a handful of windows with every field
     explained — so you can see what a "window row" actually contains.
  2. Saves a follow-along figure: episodes vs. windows on the same timeline,
     plus a zoom on a transition so the boundary windows are obvious.
  3. Compares the three labelling rules (how many windows they disagree on).
  4. Builds the windowed dataset across ALL records → data/windows_30s.csv,
     and reports the per-window class balance (what a classifier trains on).

Run:  python window_labels.py                 # default record 08219
      python window_labels.py --record 04048
      python window_labels.py --win 60        # 60 s windows instead of 30
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (list_records, window_labels, rhythm_spans, RHYTHM_COLORS,
                   DATA_DIR, FIGURES_DIR)

FOLLOW = os.path.join(FIGURES_DIR, "followalong")     # git-ignored scratch figs


def show_a_few_windows(rec, windows, fs, n=8):
    """Print raw window rows with their meaning — 'what is in the data?'."""
    print(f"\nA few 30 s windows from record {rec} "
          f"({len(windows)} windows total):\n")
    print(f"  {'#':>4}  {'start→end (s)':>16}  {'label':<8}  "
          f"{'afib_frac':>9}  pure")
    print("  " + "-" * 52)
    # Find the first transition so the sample isn't all one boring rhythm.
    flip = next((i for i in range(1, len(windows))
                 if windows["label"][i] != windows["label"][i - 1]), 0)
    lo = max(0, flip - n // 2)
    for i in range(lo, min(lo + n, len(windows))):
        w = windows[i]
        print(f"  {i:>4}  {w['start']/fs:7.0f}→{w['end']/fs:<7.0f}  "
              f"{w['label']:<8}  {w['afib_frac']:>9.2f}  {w['pure']}")
    print("\n  start/end : window bounds in samples (÷250 Hz = seconds)")
    print("  label     : rhythm propagated to this window (rule='majority')")
    print("  afib_frac : how much of the window is AFib (0=none, 1=all)")
    print("  pure      : True  → window sits inside ONE episode (clean label)")
    print("              False → window straddles an episode boundary (mixed)")


def figure_episodes_vs_windows(rec, windows, fs):
    """Follow-along: the same record as expert episodes AND as windows."""
    spans, _ = rhythm_spans(rec)
    hrs = lambda s: s / fs / 3600

    fig, ax = plt.subplots(3, 1, figsize=(13, 7),
                           gridspec_kw={"height_ratios": [1, 1, 2]})

    # Row 1 — ground-truth episodes (variable length).
    for s, e, nm in spans:
        ax[0].axvspan(hrs(s), hrs(e), color=RHYTHM_COLORS.get(nm, "0.6"))
    ax[0].set_title("Ground truth: expert rhythm episodes (variable length)")
    ax[0].set_yticks([]); ax[0].set_xlim(hrs(spans[0][0]), hrs(spans[-1][1]))

    # Row 2 — our 30 s windows, each a thin cell colored by its label.
    for w in windows:
        ax[1].axvspan(hrs(w["start"]), hrs(w["end"]),
                      color=RHYTHM_COLORS.get(w["label"], "0.6"))
    # mark the impure (boundary) windows with a tick underneath
    imp = windows[~windows["pure"]]
    ax[1].plot(hrs((imp["start"] + imp["end"]) / 2),
               np.zeros(len(imp)), "k|", ms=8)
    ax[1].set_title("Our 30 s windows (majority label; | = boundary/mixed window)")
    ax[1].set_yticks([]); ax[1].set_xlim(*ax[0].get_xlim())
    ax[1].set_xlabel("Time (h)")

    # Row 3 — zoom on the first Normal→AFib transition to see boundary windows.
    flip = next((i for i in range(1, len(windows))
                 if windows["label"][i] != windows["label"][i - 1]), None)
    if flip is not None:
        c = (windows["start"][flip] + windows["end"][flip]) / 2
        half = 6 * (windows["end"][0] - windows["start"][0])     # ±6 windows
        z0, z1 = c - half, c + half
        for s, e, nm in spans:
            ax[2].axvspan(max(s, z0) / fs, min(e, z1) / fs,
                          color=RHYTHM_COLORS.get(nm, "0.6"), alpha=0.25)
        for w in windows:
            if w["end"] >= z0 and w["start"] <= z1:
                ax[2].axvspan(w["start"] / fs, w["end"] / fs, fill=False,
                              edgecolor="k", lw=0.8)
                ax[2].text((w["start"] + w["end"]) / 2 / fs, 0.5,
                           w["label"], rotation=90, ha="center", va="center",
                           fontsize=8)
        ax[2].set_xlim(z0 / fs, z1 / fs)
        ax[2].set_title("Zoom on a transition — shaded = true rhythm, "
                        "boxes = 30 s windows")
        ax[2].set_yticks([]); ax[2].set_xlabel("Time (s)")

    plt.suptitle(f"Record {rec}: episodes → 30 s windows", fontweight="bold")
    plt.tight_layout()
    os.makedirs(FOLLOW, exist_ok=True)
    path = os.path.join(FOLLOW, f"windows_{rec}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_rules(rec, win_s):
    """How much does the boundary-labelling choice actually matter?"""
    maj, _ = window_labels(rec, win_s, rule="majority")
    anyy, _ = window_labels(rec, win_s, rule="afib_if_any")
    pure, _ = window_labels(rec, win_s, rule="pure")
    n = len(maj)
    n_mixed = int((~maj["pure"]).sum())
    n_diff = int((maj["label"] != anyy["label"]).sum())
    print(f"\nLabelling-rule comparison on {rec} ({n} windows):")
    print(f"  boundary (mixed) windows         : {n_mixed:4d}  "
          f"({100*n_mixed/n:.1f}%)")
    print(f"  majority vs afib_if_any disagree : {n_diff:4d}  "
          f"(these are the borderline AFib windows)")
    print(f"  windows 'pure' rule would drop   : "
          f"{int((pure['label']=='Mixed').sum()):4d}")
    print("  → boundary windows are few, but they're exactly the ambiguous")
    print("    onset/offset cases; the rule you pick is a recall/precision dial.")


def build_dataset(records, win_s, rule="majority"):
    """Window every record and write one flat CSV — the model training table."""
    out = os.path.join(DATA_DIR, f"windows_{int(win_s)}s.csv")
    counts = {}
    total = 0
    with open(out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["record", "start_s", "end_s", "label", "afib_frac", "pure"])
        for rec in records:
            windows, fs = window_labels(rec, win_s, rule=rule)
            for w in windows:
                wtr.writerow([rec, f"{w['start']/fs:.1f}", f"{w['end']/fs:.1f}",
                              w["label"], f"{w['afib_frac']:.3f}", int(w["pure"])])
                counts[w["label"]] = counts.get(w["label"], 0) + 1
                total += 1
    return out, counts, total


def figure_class_balance(counts, total, win_s):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = sorted(counts, key=counts.get, reverse=True)
    vals = [counts[n] for n in names]
    ax.bar(names, vals, color=[RHYTHM_COLORS.get(n, "0.6") for n in names])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{100*v/total:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{int(win_s)} s window class balance ({total:,} windows, all records)")
    ax.set_ylabel("# windows")
    plt.tight_layout()
    os.makedirs(FOLLOW, exist_ok=True)
    path = os.path.join(FOLLOW, f"window_class_balance_{int(win_s)}s.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Window afdb into labelled windows.")
    p.add_argument("--record", default="08219", help="record to illustrate")
    p.add_argument("--win", type=float, default=30.0, help="window length (s)")
    args = p.parse_args()

    windows, fs = window_labels(args.record, args.win)
    show_a_few_windows(args.record, windows, fs)
    print(f"\nSaved {figure_episodes_vs_windows(args.record, windows, fs)}")
    compare_rules(args.record, args.win)

    print("\nBuilding the full windowed dataset across all records…")
    recs = [r for r in list_records()]
    out, counts, total = build_dataset(recs, args.win)
    print(f"  wrote {total:,} windows → {out}")
    print("  class balance:")
    for nm in sorted(counts, key=counts.get, reverse=True):
        print(f"    {nm:<14} {counts[nm]:>7,}  ({100*counts[nm]/total:5.1f}%)")
    print(f"\nSaved {figure_class_balance(counts, total, args.win)}")


if __name__ == "__main__":
    main()
