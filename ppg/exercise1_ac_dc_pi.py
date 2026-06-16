"""
Exercise 1 — AC/DC split + Perfusion Index (PPG signal-quality gating).

A PPG is a big, slow **DC** light level with a tiny pulsatile **AC** ripple
riding on it. Two zero-phase filters separate them:

  * DC  = heavy low-pass (everything below the cardiac band) -> the baseline.
  * AC  = 0.5-8 Hz band-pass -> the per-beat pulse.

The **Perfusion Index (PI)** is the size of that pulse relative to the
baseline, as a percentage:

      PI = 100 * (pulse amplitude) / (mean DC light level)

PI is the standard cheap proxy for *how usable* a PPG window is. When the
sensor lifts off the skin (or perfusion is poor), the pulse shrinks, PI
collapses, and a watch will **refuse to report a heart rate** rather than
return a confident-but-wrong number. This script reproduces that: it splits a
synthetic PPG into AC/DC, computes PI in a sliding window, and gates the
windows whose PI falls below ~0.3 %.

Run:  python exercise1_ac_dc_pi.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — save PNGs without a display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from utils import synth_ppg_quality, save_figure


# A window with PI below this (percent) is considered too weak to trust.
PI_GATE = 0.3


def perfusion_index(ac, dc):
    """Perfusion Index (%) = pulse amplitude / mean DC light level.

    The pulse amplitude is measured as the 5th-to-95th-percentile spread of the
    AC signal rather than its raw min-to-max: percentiles ignore the occasional
    spike or motion outlier, so the number reflects the *typical* beat.
    """
    pulse_amplitude = np.percentile(ac, 95) - np.percentile(ac, 5)
    return 100.0 * pulse_amplitude / np.mean(dc)


def ac_dc_pi(ppg, fs):
    """Split ``ppg`` into AC (pulse) and DC (baseline); return ``(ac, dc, pi)``."""
    # DC = heavy low-pass at 0.4 Hz — keeps only the slow baseline, below the
    # slowest plausible heart rate (~0.5 Hz = 30 bpm).
    sos_dc = butter(2, 0.4 / (fs / 2), btype="low", output="sos")
    dc = sosfiltfilt(sos_dc, ppg)

    # AC = 0.5-8 Hz band-pass — the cardiac band (30-480 bpm and harmonics),
    # zero-phase so the pulse timing isn't smeared.
    sos_ac = butter(4, [0.5 / (fs / 2), 8 / (fs / 2)], btype="band", output="sos")
    ac = sosfiltfilt(sos_ac, ppg)

    pi = perfusion_index(ac, dc)
    return ac, dc, pi


def windowed_pi(ac, dc, fs, win_s=2.0, hop_s=1.0):
    """Slide a window over the pre-split AC/DC and compute PI in each.

    The signal is filtered **once** (in ``ac_dc_pi``) and only the *measurement*
    slides — filtering each short window on its own would add edge transients.

    Returns ``(centers, pis)``: the time at each window's centre and its PI.
    """
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    centers, pis = [], []
    for start in range(0, len(ac) - win + 1, hop):
        sl = slice(start, start + win)
        centers.append((start + win / 2) / fs)
        pis.append(perfusion_index(ac[sl], dc[sl]))
    return np.array(centers), np.array(pis)


def main():
    fs = 64
    t, ppg = synth_ppg_quality(fs=fs, dur=24.0)

    # Whole-record split + a single headline PI for the full signal.
    ac, dc, pi_global = ac_dc_pi(ppg, fs)

    # Per-window PI, then gate: True = keep, False = reject.
    centers, pis = windowed_pi(ac, dc, fs)
    keep = pis >= PI_GATE

    # ── Figure: DC baseline, AC pulse, and the PI gate over time ──
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax[0].plot(t, ppg, color="0.7", lw=1, label="raw PPG")
    ax[0].plot(t, dc, color="C0", lw=2, label="DC (baseline)")
    ax[0].set_title("Raw PPG and its DC baseline")
    ax[0].set_ylabel("light (a.u.)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, ac, color="C3", lw=1)
    ax[1].axhline(0, color="0.8", lw=0.8)
    ax[1].set_title("AC component (the extracted pulse)")
    ax[1].set_ylabel("a.u.")

    ax[2].plot(centers, pis, "-o", color="C2", ms=4, label="windowed PI")
    ax[2].axhline(PI_GATE, color="k", ls="--", lw=1, label=f"gate = {PI_GATE}%")
    # Shade the rejected windows so the gated region is obvious.
    for c, k in zip(centers, keep):
        if not k:
            ax[2].axvspan(c - 0.5, c + 0.5, color="C3", alpha=0.15)
    ax[2].set_title("Perfusion Index per 2 s window — shaded = rejected")
    ax[2].set_ylabel("PI (%)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="upper right")

    plt.suptitle("Exercise 1: AC/DC Split + Perfusion Index", fontweight="bold")
    plt.tight_layout()
    path = save_figure(fig, "exercise1_ac_dc_pi.png")
    plt.close(fig)

    # ── Console summary ──
    n_reject = int((~keep).sum())
    print(f"Whole-record Perfusion Index : {pi_global:.2f} %")
    print(f"Windows                      : {len(pis)}  "
          f"({n_reject} rejected, {len(pis) - n_reject} kept)")
    print(f"PI range across windows      : {pis.min():.2f} % – {pis.max():.2f} %")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
