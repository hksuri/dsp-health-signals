"""
Exercise 1 — Butterworth band-pass filtering on synthetic PPG and ECG.

A zero-phase (forward-backward) Butterworth band-pass removes both baseline
wander (the slow <0.5 Hz drift from breathing / electrode motion) and
high-frequency noise, while preserving the morphology of the pulse / PQRST
complex. Using ``sosfiltfilt`` keeps the filter numerically stable and
introduces zero net phase delay — critical for not distorting beat timing.

Run:  python exercise1_bandpass.py
"""

import matplotlib
matplotlib.use("Agg")   # headless backend — save PNGs without a display
import matplotlib.pyplot as plt

from utils import synth_ppg, synth_ecg, bandpass, save_figure


def main():
    # Build synthetic signals
    t_ppg, ppg_raw = synth_ppg(fs=100, dur=10)
    t_ecg, ecg_raw, _ = synth_ecg(fs=360, dur=10)

    # Band-pass: PPG keeps 0.5-8 Hz, ECG keeps 0.5-40 Hz
    ppg_clean = bandpass(ppg_raw, fs=100, lo=0.5, hi=8)
    ecg_clean = bandpass(ecg_raw, fs=360, lo=0.5, hi=40)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    ax[0].plot(t_ppg, ppg_raw, color="0.7", lw=1, label="raw")
    ax[0].plot(t_ppg, ppg_clean, color="C3", lw=1.5, label="bandpass 0.5–8 Hz")
    ax[0].set_title("PPG (fs = 100 Hz)")
    ax[0].set_ylabel("a.u.")
    ax[0].legend(loc="upper right")

    ax[1].plot(t_ecg, ecg_raw, color="0.7", lw=1, label="raw")
    ax[1].plot(t_ecg, ecg_clean, color="C0", lw=1.5, label="bandpass 0.5–40 Hz")
    ax[1].set_title("ECG (fs = 360 Hz)")
    ax[1].set_ylabel("mV")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right")

    plt.suptitle("Exercise 1: Butterworth Bandpass", fontweight="bold")
    plt.tight_layout()

    path = save_figure(fig, "exercise1_bandpass.png")
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
