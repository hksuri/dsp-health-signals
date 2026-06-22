"""
Database overview — how much AFib is in afdb, and where.

Loops every record's rhythm annotations (small, fast) and answers two
questions before we touch a single sample of signal:

  1. How is recording time split across rhythms? (class balance)
  2. How is AFib distributed across records? (some are nearly all AFib,
     others barely any — AFib is paroxysmal)

Run:  python overview.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (list_records, rhythm_spans, RHYTHM_COLORS, save_figure)


def collect():
    """Per-record AFib fraction + total hours, and DB-wide time per rhythm."""
    per_record = []                  # (rec, hours, afib_fraction)
    totals = {}                      # rhythm name -> total samples
    for rec in list_records():
        spans, fs = rhythm_spans(rec)
        total = sum(e - s for s, e, _ in spans)
        afib = sum(e - s for s, e, nm in spans if nm == "AFib")
        per_record.append((rec, total / fs / 3600, afib / total if total else 0))
        for s, e, nm in spans:
            totals[nm] = totals.get(nm, 0) + (e - s)
        print(f"  {rec}:  {total/fs/3600:5.1f} h   AFib {100*afib/total:5.1f}%")
    return per_record, totals, fs


def make_figure(per_record, totals, fs):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5),
                           gridspec_kw={"width_ratios": [2, 1]})

    # Panel 1 — AFib burden per record, sorted high → low.
    per_record = sorted(per_record, key=lambda r: r[2], reverse=True)
    recs = [r[0] for r in per_record]
    fracs = [100 * r[2] for r in per_record]
    ax[0].bar(recs, fracs, color="C3")
    ax[0].set_title("AFib burden per record")
    ax[0].set_ylabel("% of record in AFib")
    ax[0].set_xlabel("record")
    ax[0].tick_params(axis="x", rotation=90, labelsize=7)

    # Panel 2 — DB-wide split of recording time across rhythms.
    names = sorted(totals, key=totals.get, reverse=True)
    hours = [totals[n] / fs / 3600 for n in names]
    colors = [RHYTHM_COLORS.get(n, "0.6") for n in names]
    ax[1].bar(names, hours, color=colors)
    ax[1].set_title(f"Total time per rhythm ({sum(hours):.0f} h)")
    ax[1].set_ylabel("hours")
    ax[1].tick_params(axis="x", rotation=20)
    for i, h in enumerate(hours):
        ax[1].text(i, h, f"{100*h/sum(hours):.0f}%", ha="center", va="bottom",
                   fontsize=9)

    plt.suptitle("MIT-BIH AFib Database — overview", fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, "overview.png")


def main():
    print("Scanning afdb rhythm annotations…")
    per_record, totals, fs = collect()
    print(f"Saved {make_figure(per_record, totals, fs)}")


if __name__ == "__main__":
    main()
