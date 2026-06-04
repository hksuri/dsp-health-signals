"""
Exercise 4 — Spectrogram of a rest -> walk -> run PPG, plus a bonus
Normalized LMS (NLMS) adaptive filter for motion-artifact rejection.

During exercise the foot-strike cadence (steps/second) shows up in the PPG as a
strong, drifting band that overlaps the cardiac frequency — you cannot separate
them with a fixed filter because both move with intensity. The trick used by
real wrist wearables is an *accelerometer reference*: an NLMS adaptive filter
learns, sample-by-sample, the mapping from the accelerometer (pure motion) to
the motion component in the PPG, then subtracts it, leaving the cardiac signal.

Two figures are produced:
  * exercise4_timedomain.png  — raw PPG, accelerometer reference, NLMS output
  * exercise4_spectrogram.png — spectrograms of corrupted / cleaned / ground-truth

Run:  python exercise4_spectrogram_nlms.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from utils import synth_ppg_motion, lowpass, save_figure


# ── Normalized LMS adaptive filter ────────────────────────────
def nlms(primary, ref, mu=0.02, L=32, eps=1e-6):
    """
    Normalized LMS — removes from `primary` whatever is correlated with `ref`
    (the accelerometer signal).

    primary : corrupted PPG signal
    ref     : accelerometer reference (motion proxy)
    mu      : step size (0 < mu < 2 for stability)
    L       : filter length (taps)
    eps     : small constant to prevent division by zero

    Returns the error signal e[n] = primary[n] - y_hat[n], which is the
    motion-rejected PPG.
    """
    N = len(primary)
    w = np.zeros(L)          # filter weights, initialised to zero
    e = np.zeros(N)          # output: cleaned signal

    for n in range(N):
        # Build input vector: last L samples of reference
        # (pad with zeros before the signal starts)
        x = np.array([ref[n - k] if (n - k) >= 0 else 0.0
                      for k in range(L)])

        # Filter output: dot product of weights and input vector
        y_hat = np.dot(w, x)

        # Error: what primary has that the motion model can't explain
        e[n] = primary[n] - y_hat

        # Normalised weight update: divide by signal power so step size is
        # self-scaling regardless of input amplitude
        power = np.dot(x, x) + eps
        w += (mu / power) * e[n] * x

    return e


# ── Spectrogram helper ────────────────────────────────────────
def plot_spectrogram(ax, sig, fs, title, vmin=-40, vmax=10):
    f, t_s, Sxx = spectrogram(sig, fs=fs, window="hann",
                              nperseg=256,    # 4 s window at 64 Hz
                              noverlap=192)   # 75% overlap
    db = 10 * np.log10(Sxx + 1e-12)
    mesh = ax.pcolormesh(t_s, f, db, shading="gouraud",
                         cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    # Phase markers
    for boundary in (20, 40):
        ax.axvline(boundary, color="white", lw=1, ls="--", alpha=0.6)
    ax.text(5, 7.4, "rest", color="white", fontsize=9, alpha=0.8)
    ax.text(25, 7.4, "walk", color="white", fontsize=9, alpha=0.8)
    ax.text(45, 7.4, "run", color="white", fontsize=9, alpha=0.8)
    return mesh


def main():
    fs = 64
    t, ppg_raw, acc, ppg_clean = synth_ppg_motion(fs=fs, dur=60.0)

    # Run NLMS to remove motion artifact, then a light low-pass to clean
    # residual high-frequency noise.
    ppg_nlms = nlms(ppg_raw, acc, mu=0.02, L=32)
    ppg_nlms_filt = lowpass(ppg_nlms, fs)

    # ── Figure 1: time domain overview ───────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(t, ppg_raw, color="0.6", lw=0.7, label="raw PPG (corrupted)")
    axes[0].plot(t, ppg_clean, color="C3", lw=1.2,
                 label="clean cardiac (ground truth)", alpha=0.8)
    axes[0].set_ylabel("a.u.")
    axes[0].set_title("PPG signals")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(t, acc, color="C1", lw=0.8, label="accelerometer (motion reference)")
    axes[1].set_ylabel("a.u.")
    axes[1].set_title("Accelerometer reference fed to NLMS")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(t, ppg_nlms_filt, color="C2", lw=1.0, label="NLMS output (motion rejected)")
    axes[2].plot(t, ppg_clean, color="C3", lw=1.2,
                 label="clean cardiac (ground truth)", alpha=0.6, ls="--")
    axes[2].set_ylabel("a.u.")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("After NLMS adaptive filtering")
    axes[2].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.axvline(20, color="gray", lw=1, ls="--", alpha=0.5)
        ax.axvline(40, color="gray", lw=1, ls="--", alpha=0.5)
        ax.text(3, ax.get_ylim()[1] * 0.85, "rest", fontsize=8, color="gray")
        ax.text(23, ax.get_ylim()[1] * 0.85, "walk", fontsize=8, color="gray")
        ax.text(43, ax.get_ylim()[1] * 0.85, "run", fontsize=8, color="gray")

    plt.suptitle("Exercise 4 + Bonus: PPG Motion Artifact Rejection", fontweight="bold")
    plt.tight_layout()
    path1 = save_figure(fig, "exercise4_timedomain.png")
    print(f"Saved {path1}")
    plt.close(fig)

    # ── Figure 2: spectrograms stacked ───────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    m1 = plot_spectrogram(axes[0], ppg_raw, fs,
                          "Corrupted PPG — cardiac + cadence bands visible")
    plot_spectrogram(axes[1], ppg_nlms_filt, fs,
                     "After NLMS — cadence band suppressed, HR line survives")
    plot_spectrogram(axes[2], ppg_clean, fs,
                     "Ground truth clean cardiac")

    axes[2].set_xlabel("Time (s)")

    # Shared colourbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(m1, cax=cbar_ax, label="Power (dB)")

    plt.suptitle("Exercise 4: Spectrogram — Rest → Walk → Run\n"
                 "Bonus: NLMS adaptive motion rejection", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    path2 = save_figure(fig, "exercise4_spectrogram.png")
    print(f"Saved {path2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
