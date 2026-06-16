"""
Exercise 3 — 50/60 Hz notch filter for powerline interference.

Mains hum (60 Hz in North America, 50 Hz elsewhere) couples into biopotential
recordings as a narrow, persistent spectral spike. A zero-phase IIR notch
filter (``iirnotch`` + ``filtfilt``) surgically removes it while leaving the
surrounding ECG spectrum — and the beat morphology — intact. We verify the
attenuation directly with a Welch power spectral density (PSD) estimate before
and after filtering.

Run:  python exercise3_notch.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import iirnotch, filtfilt, welch

from utils import synth_ecg, bandpass, save_figure


# ── Notch filter ──────────────────────────────────────────────
def notch(sig, fs, f0=60.0, Q=30):
    """Remove power-line interference. Q sets the notch width."""
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, sig)   # zero-phase: protect morphology


def main():
    fs = 360
    t, ecg_raw, _ = synth_ecg(fs=fs, dur=10.0)

    ecg_clean = bandpass(ecg_raw, fs=fs, lo=0.5, hi=40)
    ecg_notched = notch(ecg_clean, fs=fs, f0=60)

    # Welch PSD before and after
    f_ax, psd_before = welch(ecg_clean, fs=fs, nperseg=1024)
    f_ax, psd_after = welch(ecg_notched, fs=fs, nperseg=1024)

    psd_before_db = 10 * np.log10(psd_before + 1e-12)
    psd_after_db = 10 * np.log10(psd_after + 1e-12)

    idx_60 = np.argmin(np.abs(f_ax - 60))
    drop_db = psd_before_db[idx_60] - psd_after_db[idx_60]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Panel 1: time domain — raw vs bandpassed vs notched
    axes[0].plot(t, ecg_raw, color="0.7", lw=0.8, label="raw (with 60 Hz + wander)")
    axes[0].plot(t, ecg_clean, color="C0", lw=1.2, label="bandpass 0.5–40 Hz", alpha=0.8)
    axes[0].plot(t, ecg_notched, color="C3", lw=1.2, label="+ notch 60 Hz", alpha=0.9)
    axes[0].set_ylabel("mV")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Time domain: raw → bandpass → notch")
    axes[0].legend(loc="upper right")

    # Panel 2: PSD before vs after (full spectrum)
    axes[1].plot(f_ax, psd_before_db, color="C0", lw=1.5, label="before notch")
    axes[1].plot(f_ax, psd_after_db, color="C3", lw=1.5, label="after notch")
    axes[1].axvline(60, color="gray", lw=1, ls="--", label="60 Hz")
    axes[1].set_ylabel("Power (dB)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title(f"Welch PSD — 60 Hz spike drops {drop_db:.1f} dB after notch")
    axes[1].legend(loc="upper right")

    # Panel 3: zoomed PSD around 60 Hz to show notch depth clearly
    zoom_mask = (f_ax >= 50) & (f_ax <= 70)
    axes[2].plot(f_ax[zoom_mask], psd_before_db[zoom_mask], color="C0", lw=2, label="before notch")
    axes[2].plot(f_ax[zoom_mask], psd_after_db[zoom_mask], color="C3", lw=2, label="after notch")
    axes[2].axvline(60, color="gray", lw=1, ls="--", label="60 Hz")
    axes[2].set_ylabel("Power (dB)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_title("Zoomed 50–70 Hz: notch depth")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    path = save_figure(fig, "exercise3_notch.png")
    print(f"Saved {path}")
    plt.close(fig)

    print(f"60 Hz power before: {psd_before_db[idx_60]:.1f} dB")
    print(f"60 Hz power after:  {psd_after_db[idx_60]:.1f} dB")
    print(f"Attenuation:        {drop_db:.1f} dB")


if __name__ == "__main__":
    main()
