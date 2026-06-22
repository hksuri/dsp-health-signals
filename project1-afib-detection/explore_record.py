"""
Single-record exploration — what does AFib actually look like?

Pulls one afdb record and shows the signal-level story behind AFib detection:

  • two ECG strips, normal rhythm vs AFib (no P-waves, irregular spacing)
  • a Poincaré plot of RR intervals — the classic AFib signature: a tight
    cluster for normal rhythm, a diffuse cloud for AFib ("irregularly irregular")
  • the RR tachogram across the whole record, with AFib episodes shaded

Play with it:  python explore_record.py --record 04048
               python explore_record.py --record 08455 --channel 1
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (rhythm_spans, beats_with_rhythm, rr_intervals,
                   load_signal, find_episode, RHYTHM_COLORS, save_figure)


def ecg_strip(ax, rec, span, name, seconds=6.0, channel=0):
    """Plot a few seconds of ECG taken from inside a rhythm episode."""
    if span is None:
        ax.text(0.5, 0.5, f"no {name} episode", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
        return
    s, e = span
    mid = (s + e) // 2                       # sample a window from the middle
    sig, fs, ch = load_signal(rec, mid, mid + int(seconds * 250), channel)
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color=RHYTHM_COLORS.get(name, "0.3"), lw=0.8)
    ax.set_title(f"{name} — {ch}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mV")


def poincare(ax, rr, labels):
    """RR_n vs RR_{n+1}, colored by rhythm — the AFib fingerprint."""
    x, y, lab = rr[:-1], rr[1:], labels[1:]
    # Draw the diffuse AFib cloud first, then the tight Normal cluster on top.
    for name in ("AFib", "Normal"):
        m = lab == name
        if m.any():
            ax.scatter(x[m], y[m], s=3, alpha=0.25,
                       color=RHYTHM_COLORS[name], label=f"{name} (n={m.sum()})")
    ax.set_title("Poincaré plot of RR intervals")
    ax.set_xlabel("RRₙ (s)"); ax.set_ylabel("RRₙ₊₁ (s)")
    ax.set_aspect("equal")
    leg = ax.legend(loc="upper left", fontsize=8)
    for h in leg.legend_handles:
        h.set_alpha(1)


def tachogram(ax, rec, beats, fs, rr, keep, spans):
    """RR over the whole record, with AFib episodes shaded."""
    rr_t = beats[1:][keep] / fs / 3600        # hours
    ax.plot(rr_t, rr[keep], color="0.4", lw=0.4)
    for s, e, nm in spans:
        if nm == "AFib":
            ax.axvspan(s / fs / 3600, e / fs / 3600,
                       color=RHYTHM_COLORS["AFib"], alpha=0.15)
    ax.set_title("RR tachogram (AFib episodes shaded)")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("RR (s)")


def main():
    p = argparse.ArgumentParser(description="Explore one afdb record.")
    p.add_argument("--record", default="04015", help="afdb record id (default 04015)")
    p.add_argument("--channel", type=int, default=0, help="ECG channel 0 or 1")
    args = p.parse_args()
    rec = args.record

    spans, fs = rhythm_spans(rec)
    beats, fs, labels = beats_with_rhythm(rec)
    rr, keep = rr_intervals(beats, fs)
    rr_labels = labels[1:]                     # label each interval by its 2nd beat

    afib_pct = 100 * sum(e - s for s, e, nm in spans if nm == "AFib") \
        / sum(e - s for s, e, _ in spans)
    print(f"Record {rec}:  {len(beats)} beats, {afib_pct:.1f}% AFib, "
          f"{len(spans)} rhythm episodes")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ecg_strip(ax[0, 0], rec, find_episode(spans, "Normal", 8, fs), "Normal",
              channel=args.channel)
    ecg_strip(ax[0, 1], rec, find_episode(spans, "AFib", 8, fs), "AFib",
              channel=args.channel)
    poincare(ax[1, 0], rr[keep], rr_labels[keep])
    tachogram(ax[1, 1], rec, beats, fs, rr, keep, spans)

    plt.suptitle(f"afdb record {rec} — {afib_pct:.0f}% AFib", fontweight="bold")
    plt.tight_layout()
    print(f"Saved {save_figure(fig, f'record_{rec}.png')}")


if __name__ == "__main__":
    main()
