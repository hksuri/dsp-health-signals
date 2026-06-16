"""
Exercise 2 — Pan-Tompkins QRS detection from scratch.

Implements the classic Pan-Tompkins pipeline (band-pass -> derivative -> square
-> moving-window integration -> adaptive threshold -> peak backtracking) and
validates it two ways:

  1. On a synthetic ECG with known beat positions.
  2. On MIT-BIH Arrhythmia Database record 100 (downloaded from PhysioNet via
     ``wfdb``), scored against the cardiologist annotations.

Detection quality is reported as Sensitivity (Se = TP / (TP + FN)) and
Positive Predictivity (+P = TP / (TP + FP)) within a ±50 ms matching tolerance.

If ``wfdb`` is missing or PhysioNet is unreachable, the MIT-BIH section is
skipped gracefully and only the synthetic-data figure is produced.

Run:  python exercise2_pan_tompkins.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

from utils import synth_ecg, save_figure


# ── Pan-Tompkins QRS detector ─────────────────────────────────
def pan_tompkins(ecg, fs):
    # 1. Bandpass 5-15 Hz (QRS energy band)
    nyq = fs / 2
    b, a = butter(3, [5 / nyq, 15 / nyq], btype="band")
    x = filtfilt(b, a, ecg)

    # 2. Derivative — emphasize steep QRS slopes
    d = np.diff(x, prepend=x[0])

    # 3. Square — make everything positive, boost big slopes
    sq = d ** 2

    # 4. Moving-window integration (~150 ms)
    w = int(0.150 * fs)
    mwi = np.convolve(sq, np.ones(w) / w, mode="same")

    # 5. Adaptive threshold + peak picking
    thr = 0.15 * np.percentile(mwi, 98)
    mwi_peaks = []
    for i in range(1, len(mwi) - 1):
        if mwi[i] > thr and mwi[i] > mwi[i - 1] and mwi[i] >= mwi[i + 1]:
            if not mwi_peaks or (i - mwi_peaks[-1]) > 0.25 * fs:
                mwi_peaks.append(i)

    # 6. Backtrack: find true R-peak in filtered signal within ±150 ms
    search_r = int(0.150 * fs)
    r_peaks = []
    for p in mwi_peaks:
        lo = max(0, p - search_r)
        hi = min(len(x), p + search_r)
        r_peaks.append(lo + np.argmax(x[lo:hi]))

    return np.array(r_peaks), x, mwi


def score(detected, reference, fs, tol_ms=50):
    """Sensitivity and positive predictivity within ±tol_ms."""
    tol = int(tol_ms * fs / 1000)
    ref = np.array(reference)
    matched_ref = np.zeros(len(ref), dtype=bool)
    tp = 0
    for d in detected:
        diffs = np.abs(ref - d)
        j = np.argmin(diffs)
        if diffs[j] <= tol and not matched_ref[j]:
            tp += 1
            matched_ref[j] = True
    fn = len(ref) - tp
    fp = len(detected) - tp
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return dict(TP=tp, FN=fn, FP=fp, Se=se, PP=pp)


def _diagnostic_plot(ecg_sig, fs, detected, x_bp, mwi, title, filename,
                     ref=None, n_show=None):
    """Three-panel Pan-Tompkins diagnostic, saved to figures/."""
    thr = 0.15 * np.percentile(mwi, 98)
    if n_show is None:
        n_show = len(ecg_sig)
    n_show = min(n_show, len(ecg_sig))
    t = np.arange(n_show) / fs

    det_in = detected[detected < n_show]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel 1: raw ECG + detected peaks (+ annotations if provided)
    axes[0].plot(t, ecg_sig[:n_show], color="0.6", lw=0.8, label="raw ECG")
    axes[0].scatter(t[det_in], x_bp[det_in], color="red", s=40, zorder=5,
                    label="detected R")
    if ref is not None:
        ref_in = ref[ref < n_show]
        axes[0].scatter(t[ref_in], ecg_sig[ref_in], color="lime", s=25,
                        zorder=4, marker="x", linewidths=1.5,
                        label="annotated R")
    axes[0].set_ylabel("mV")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")

    # Panel 2: bandpass output
    axes[1].plot(t, x_bp[:n_show], color="C0", lw=1, label="bandpass 5–15 Hz")
    axes[1].set_ylabel("mV")
    axes[1].set_title("After bandpass")
    axes[1].legend(loc="upper right")

    # Panel 3: MWI + threshold
    axes[2].plot(t, mwi[:n_show], color="C2", lw=1, label="MWI (energy envelope)")
    axes[2].axhline(thr, color="red", lw=1, ls="--", label=f"threshold ({thr:.5f})")
    axes[2].scatter(t[det_in], mwi[det_in], color="red", s=40, zorder=5)
    axes[2].set_ylabel("energy")
    axes[2].set_title("Moving-window integrator + threshold")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    path = save_figure(fig, filename)
    print(f"Saved {path}")
    plt.close(fig)


def run_synthetic():
    fs = 360
    _, ecg_raw, ref_syn = synth_ecg(fs=fs, dur=10.0)
    detected, x_bp, mwi = pan_tompkins(ecg_raw, fs)

    res = score(detected, ref_syn, fs=fs, tol_ms=50)
    print(f"Synthetic   →  Se={res['Se']:.3f}  +P={res['PP']:.3f}  "
          f"TP={res['TP']}  FP={res['FP']}  FN={res['FN']}")

    _diagnostic_plot(ecg_raw, fs, detected, x_bp, mwi,
                     "Synthetic — Raw ECG + detected R-peaks",
                     "exercise2_pan_tompkins_synthetic.png")


def run_mitbih():
    """Validate on MIT-BIH record 100. Returns True on success."""
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed — skipping MIT-BIH (pip install wfdb)")
        return False

    try:
        rec = wfdb.rdrecord("100", pn_dir="mitdb")
        ann = wfdb.rdann("100", "atr", pn_dir="mitdb")
    except Exception as exc:  # network / PhysioNet unavailable
        print(f"Could not download MIT-BIH record 100 ({exc}) — "
              "skipping real-data validation, synthetic figure still produced.")
        return False

    ecg = rec.p_signal[:, 0]
    fs = rec.fs

    beat_labels = set("NLRBAaJSVrFejnE/fQ")
    ref = np.array([s for s, c in zip(ann.sample, ann.symbol)
                    if c in beat_labels])

    detected, x_bp, mwi = pan_tompkins(ecg, fs)
    res = score(detected, ref, fs, tol_ms=50)
    print(f"MIT-BIH 100 →  Se={res['Se']:.3f}  +P={res['PP']:.3f}  "
          f"TP={res['TP']}  FP={res['FP']}  FN={res['FN']}")

    _diagnostic_plot(ecg, fs, detected, x_bp, mwi,
                     "MIT-BIH record 100 — raw + detections vs annotations",
                     "exercise2_pan_tompkins_mitbih.png",
                     ref=ref, n_show=10 * fs)   # first 10 seconds
    return True


def main():
    run_synthetic()
    run_mitbih()


if __name__ == "__main__":
    main()
