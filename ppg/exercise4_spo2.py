"""
Exercise 4 — SpO₂ via ratio-of-ratios.

A pulse oximeter shines two wavelengths through tissue — red (~660 nm) and
infrared (~940 nm) — because oxygenated and deoxygenated haemoglobin absorb
them differently. It never measures absolute absorption; it measures the
*pulsatile* (AC) signal relative to the steady (DC) light level at each
wavelength, then takes the ratio of those ratios:

    R = (AC_red / DC_red) / (AC_ir / DC_ir)              (eq 4.1)
    SpO₂ ≈ 110 − 25·R                                    (eq 4.2)

The magic is that R is **self-normalizing**: dividing AC by DC at each
wavelength cancels skin tone, sensor gain, contact pressure, and how hard the
finger is pressed — anything that scales the whole channel. Only the
oxygen-dependent ratio survives. This exercise shows exactly that, two ways:

  1. **Accuracy sweep** — synthesize red+IR PPG at known SpO₂ from 80–100 % and
     check the estimate tracks truth.
  2. **Self-normalization** — hold SpO₂ fixed and vary perfusion + DC light
     level over a wide range; the raw AC amplitude swings ~85× (perfusion ×
     brightness), but the SpO₂ estimate barely moves (±0.8 %).

Honesty note: the synthetic data is built with the *same* 110/25 calibration
the estimator inverts, so this validates the **signal processing**, not the
calibration curve. Those constants are empirical (fit to human cohorts), not
physics — a real device needs clinical calibration. The point here is the DSP:
R extraction is robust because it is a ratio.

Run:  python exercise4_spo2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from utils import synth_ppg_red_ir, save_figure, SPO2_CAL_A, SPO2_CAL_B


# ── Ratio-of-ratios SpO₂ ──────────────────────────────────────
def spo2(red, ir, fs):
    """Estimate SpO₂ from red + IR PPG via the ratio-of-ratios (eqs 4.1, 4.2).

    Returns ``(spo2_pct, R)``.
    """
    sos = butter(4, [0.5 / (fs / 2), 8 / (fs / 2)], btype="band", output="sos")
    ac_red = sosfiltfilt(sos, red)         # pulsatile component, each channel
    ac_ir = sosfiltfilt(sos, ir)
    dc_red, dc_ir = np.mean(red), np.mean(ir)   # steady light level

    # Robust AC amplitude (p95−p5 ignores the odd motion spike, like the PI).
    acr = np.percentile(ac_red, 95) - np.percentile(ac_red, 5)
    aci = np.percentile(ac_ir, 95) - np.percentile(ac_ir, 5)

    R = (acr / dc_red) / (aci / dc_ir)     # ratio of ratios — self-normalizing
    return SPO2_CAL_A - SPO2_CAL_B * R, R


# ── Experiments ───────────────────────────────────────────────
def accuracy_sweep(fs=64):
    """Estimate SpO₂ across a sweep of known true values."""
    truths = np.arange(80, 101, 2)
    ests = np.array([spo2(*synth_ppg_red_ir(s, fs=fs, seed=i)[1:], fs)[0]
                     for i, s in enumerate(truths)])
    mae = np.mean(np.abs(ests - truths))
    print(f"Accuracy sweep   80–100 %:  MAE = {mae:.2f} %  "
          f"(max |err| = {np.max(np.abs(ests - truths)):.2f} %)")
    return truths, ests, mae


def self_norm_sweep(fs=64, spo2_fixed=97.0):
    """Hold SpO₂ fixed, vary perfusion + DC light level; track estimate vs raw AC."""
    scales = np.linspace(0.3, 3.0, 12)        # 10× span in perfusion / brightness
    ests, ac_amps = [], []
    for i, k in enumerate(scales):
        t, red, ir = synth_ppg_red_ir(spo2_fixed, fs=fs,
                                      dc_ir=8000.0 * k, dc_red=6000.0 * k,
                                      perfusion=0.02 * k, seed=100 + i)
        est, _ = spo2(red, ir, fs)
        ests.append(est)
        sos = butter(4, [0.5 / (fs / 2), 8 / (fs / 2)], btype="band", output="sos")
        ac = sosfiltfilt(sos, ir)
        ac_amps.append(np.percentile(ac, 95) - np.percentile(ac, 5))
    ests, ac_amps = np.array(ests), np.array(ac_amps)
    print(f"Self-normalization (true {spo2_fixed:.0f} %): raw IR AC swings "
          f"{ac_amps.max() / ac_amps.min():.1f}×, but SpO₂ estimate stays "
          f"{ests.min():.1f}–{ests.max():.1f} % "
          f"(±{(ests.max() - ests.min()) / 2:.2f} %)")
    return scales, ests, ac_amps, spo2_fixed


def make_figure(fs=64):
    sweep = accuracy_sweep(fs)
    norm = self_norm_sweep(fs)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1 — example red + IR waveforms (one SpO₂), AC ripple on DC.
    t, red, ir = synth_ppg_red_ir(97, fs=fs, dur=6.0, seed=7)
    ax[0].plot(t, ir, color="C3", lw=1.0, label="IR (~940 nm)")
    ax[0].plot(t, red, color="C1", lw=1.0, label="red (~660 nm)")
    ax[0].set_title("Two-wavelength PPG (SpO₂ = 97 %)")
    ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("light level (a.u.)")
    ax[0].legend(loc="upper right", fontsize=8)

    # Panel 2 — estimated vs true across the sweep.
    truths, ests, mae = sweep
    ax[1].plot(truths, truths, color="0.6", ls="--", lw=1, label="ideal (y = x)")
    ax[1].plot(truths, ests, "o-", color="C0", lw=1.3, label="estimate")
    ax[1].set_title(f"Accuracy across 80–100 %  (MAE {mae:.2f} %)")
    ax[1].set_xlabel("true SpO₂ (%)"); ax[1].set_ylabel("estimated SpO₂ (%)")
    ax[1].legend(loc="upper left", fontsize=8)

    # Panel 3 — self-normalization: estimate flat while raw AC amplitude soars.
    scales, ne, ac_amps, sfix = norm
    ln1 = ax[2].plot(scales, ne, "o-", color="C0", lw=1.3,
                     label="SpO₂ estimate")
    ax[2].axhline(sfix, color="0.6", ls="--", lw=1, label=f"true {sfix:.0f} %")
    ax[2].set_ylim(sfix - 5, sfix + 5)
    ax[2].set_title("Self-normalizing: estimate ignores perfusion / brightness")
    ax[2].set_xlabel("perfusion + light-level scale (×)")
    ax[2].set_ylabel("estimated SpO₂ (%)", color="C0")
    ax2b = ax[2].twinx()
    ln2 = ax2b.plot(scales, ac_amps, "s--", color="C3", lw=1.1,
                    label="raw IR AC amplitude")
    ax2b.set_ylabel("raw IR AC amplitude (a.u.)", color="C3")
    lns = ln1 + ln2
    ax[2].legend(lns, [l.get_label() for l in lns], loc="lower right", fontsize=8)

    plt.suptitle("Exercise 4: SpO₂ via Ratio-of-Ratios", fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, "exercise4_spo2.png")


def main():
    path = make_figure()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
