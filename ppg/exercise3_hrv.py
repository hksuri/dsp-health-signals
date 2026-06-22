"""
Exercise 3 — HRV features from a tachogram.

Heart-rate *variability* is the structure in the beat-to-beat intervals (the
"tachogram"), not the average rate. This computes the standard clinical HRV
features two ways and validates each:

  Time domain   — SDNN, RMSSD, pNN50 (eqs 3.1-3.3).
  Frequency dom — LF (0.04-0.15 Hz) / HF (0.15-0.40 Hz) power and their ratio
                  (eq 3.4), via Welch PSD of the *resampled* tachogram.

The catch the frequency-domain step exists to teach: a tachogram is sampled
once per beat, so its time axis is **uneven**. An FFT assumes a uniform grid,
so the intervals must first be cubic-spline interpolated onto an even grid
(here 4 Hz) before any spectral estimate is meaningful.

It runs on two sources:
  1. A synthetic tachogram with KNOWN injected LF/HF oscillations, so the
     spectral peaks can be checked against the frequencies we put in.
  2. Real NN intervals from MIT-BIH record 100 (downloaded via ``wfdb``),
     scored for the same features. Skipped gracefully if wfdb / PhysioNet
     is unavailable.

Run:  python exercise3_hrv.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.signal import welch

from utils import (synth_tachogram, load_mitbih_nn, save_figure,
                   HRV_LF_BAND, HRV_HF_BAND)


# ── HRV features ──────────────────────────────────────────────
def hrv_time(nn_ms):
    """Time-domain HRV from NN intervals (ms): SDNN, RMSSD, pNN50."""
    diff = np.diff(nn_ms)                          # successive-interval changes
    sdnn = np.std(nn_ms)                           # eq 3.1 — overall variability
    rmssd = np.sqrt(np.mean(diff ** 2))            # eq 3.2 — short-term (vagal)
    pnn50 = 100.0 * np.mean(np.abs(diff) > 50)     # eq 3.3 — % of jumps > 50 ms
    return dict(sdnn=sdnn, rmssd=rmssd, pnn50=pnn50)


def hrv_freq(nn_ms, fs_interp=4.0):
    """LF/HF power from the tachogram, resampled onto an even grid first.

    Returns a dict with lf, hf, lf_hf plus the (f, psd) arrays so the caller
    can plot the spectrum.
    """
    # Beat times = cumulative sum of intervals (this is the *uneven* grid).
    t = np.cumsum(nn_ms) / 1000.0                  # seconds
    # Uniform time grid + cubic-spline resample — the step everyone forgets.
    t_even = np.arange(t[0], t[-1], 1.0 / fs_interp)
    nn_even = CubicSpline(t, nn_ms)(t_even)

    # Welch PSD of the detrended (mean-removed) tachogram.
    nperseg = min(256, len(nn_even))
    f, psd = welch(nn_even - nn_even.mean(), fs=fs_interp, nperseg=nperseg)

    lf = _band_power(f, psd, HRV_LF_BAND)
    hf = _band_power(f, psd, HRV_HF_BAND)
    return dict(lf=lf, hf=hf, lf_hf=lf / hf if hf > 0 else np.nan, f=f, psd=psd)


def _band_power(f, psd, band):
    """Integrate the PSD over [lo, hi) with the trapezoid rule."""
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return trapezoid(psd[m], f[m])


# ── Plotting ──────────────────────────────────────────────────
def _plot_tachogram(ax, nn_ms, title):
    beat_t = np.cumsum(nn_ms) / 1000.0
    ax.plot(beat_t, nn_ms, color="C0", lw=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NN interval (ms)")


def _plot_psd(ax, fr, title, mark_hz=None):
    f, psd = fr["f"], fr["psd"]
    ax.semilogy(f, psd, color="0.35", lw=1.1)
    for band, color, name in ((HRV_LF_BAND, "C0", "LF"),
                              (HRV_HF_BAND, "C3", "HF")):
        ax.axvspan(*band, color=color, alpha=0.15)
        mid = sum(band) / 2
        ax.text(mid, ax.get_ylim()[1], name, color=color,
                ha="center", va="top", fontsize=9, fontweight="bold")
    if mark_hz:                                    # injected ground-truth peaks
        for hz in mark_hz:
            ax.axvline(hz, color="green", ls="--", lw=1,
                       label=f"injected {hz:.2f} Hz")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (ms²/Hz)")


def make_figure(syn, mit):
    """2×2 grid: synthetic (top) and real MIT-BIH (bottom) — tachogram + PSD."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    nn_s, td_s, fd_s, meta = syn
    _plot_tachogram(ax[0, 0], nn_s,
                    f"Synthetic tachogram — SDNN {td_s['sdnn']:.1f} ms, "
                    f"RMSSD {td_s['rmssd']:.1f} ms")
    _plot_psd(ax[0, 1], fd_s,
              f"Synthetic PSD — LF/HF {fd_s['lf_hf']:.2f}",
              mark_hz=(meta["lf_hz"], meta["hf_hz"]))

    if mit is not None:
        nn_m, td_m, fd_m = mit
        _plot_tachogram(ax[1, 0], nn_m,
                        f"MIT-BIH 100 tachogram — SDNN {td_m['sdnn']:.1f} ms, "
                        f"RMSSD {td_m['rmssd']:.1f} ms")
        _plot_psd(ax[1, 1], fd_m, f"MIT-BIH 100 PSD — LF/HF {fd_m['lf_hf']:.2f}")
    else:
        for a in (ax[1, 0], ax[1, 1]):
            a.text(0.5, 0.5, "MIT-BIH unavailable\n(install wfdb / network)",
                   ha="center", va="center", transform=a.transAxes, color="0.5")
            a.set_xticks([]); a.set_yticks([])

    plt.suptitle("Exercise 3: HRV Features from a Tachogram", fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, "exercise3_hrv.png")


# ── Runs ──────────────────────────────────────────────────────
def _print_row(label, td, fd):
    print(f"{label:<14}"
          f"SDNN={td['sdnn']:6.1f}  RMSSD={td['rmssd']:6.1f}  "
          f"pNN50={td['pnn50']:5.1f}%   "
          f"LF={fd['lf']:8.1f}  HF={fd['hf']:8.1f}  LF/HF={fd['lf_hf']:.2f}")


def run_synthetic():
    nn, meta = synth_tachogram()
    td, fd = hrv_time(nn), hrv_freq(nn)
    _print_row("Synthetic", td, fd)
    print(f"               (injected LF={meta['lf_hz']:.2f} Hz, "
          f"HF={meta['hf_hz']:.2f} Hz)")
    return nn, td, fd, meta


def run_mitbih(record="100"):
    """Returns (nn, td, fd) or None if wfdb / PhysioNet is unavailable."""
    try:
        import wfdb  # noqa: F401
    except ImportError:
        print("wfdb not installed — skipping MIT-BIH (pip install wfdb)")
        return None
    try:
        nn, _ = load_mitbih_nn(record)
    except Exception as exc:                       # network / PhysioNet down
        print(f"Could not load MIT-BIH record {record} ({exc}) — "
              "skipping real-data HRV.")
        return None
    td, fd = hrv_time(nn), hrv_freq(nn)
    _print_row(f"MIT-BIH {record}", td, fd)
    return nn, td, fd


def main():
    syn = run_synthetic()
    mit = run_mitbih()
    path = make_figure(syn, mit)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
