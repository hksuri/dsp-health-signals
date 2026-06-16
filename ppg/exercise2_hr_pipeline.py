"""
Exercise 2 — Heart-rate-from-PPG pipeline on PPG-DaLiA, with activity-stratified MAE.

The pipeline turns a wrist PPG into a heart-rate estimate:

  band-pass (0.5-4 Hz)  ->  peak detection (refractory + prominence)
        ->  inter-beat intervals  ->  instantaneous HR  ->  reject + smooth

We then score it against PPG-DaLiA's ECG-derived ground-truth HR on the same
8 s / 2 s window grid, and — this is the whole point — break the error out
**by activity** (sitting / walking / cycling / …). A wrist PPG is accurate at
rest and falls apart under motion, because cadence energy invades the cardiac
band; reporting one averaged MAE hides that. The activity-stratified table is
the honest result, and the motion rows are exactly what an accelerometer-
referenced adaptive filter (Project 2) goes on to fix.

This script uses the **real** PPG-DaLiA dataset (no synthetic fallback).
Download it (UCI ML Repository, "PPG-DaLiA", ~3 GB), unzip, and point the
script at the resulting ``PPG_FieldStudy`` folder:

    python exercise2_hr_pipeline.py --data-dir /path/to/PPG_FieldStudy
    # or set PPG_DALIA_DIR in the environment

Run:  python exercise2_hr_pipeline.py [--data-dir DIR] [--subject S3]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — save PNGs without a display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

from utils import (
    save_figure,
    load_dalia_subject,
    find_dalia_subjects,
    DALIA_ACTIVITIES,
    DALIA_FS_BVP,
    DALIA_FS_ACT,
    DALIA_HR_WIN_S,
    DALIA_HR_SHIFT_S,
)


def hr_from_ppg(ppg, fs):
    """Estimate instantaneous HR (bpm) from a PPG segment.

    Returns ``(peaks, hr_smooth)`` where ``peaks`` are sample indices of the
    detected pulses and ``hr_smooth`` is the per-beat HR after outlier removal
    and a short moving average. Faithful to the Week 3 tutorial recipe.
    """
    # 1. Band-pass to isolate the AC cardiac component (0.5-4 Hz ≈ 30-240 bpm).
    sos = butter(4, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band", output="sos")
    ac = sosfiltfilt(sos, ppg)

    # 2. Peak detection with a refractory period (min spacing → max ~200 bpm)
    #    and a prominence floor so noise wiggles don't count as beats.
    min_dist = int(0.3 * fs)                         # 0.3 s → < 200 bpm
    peaks, _ = find_peaks(ac, distance=min_dist,
                          prominence=np.std(ac) * 0.5)

    # 3. Instantaneous HR from inter-beat intervals.
    ibi = np.diff(peaks) / fs                        # seconds between beats
    hr = 60.0 / ibi if ibi.size else np.array([])    # bpm

    # 4. Reject physiologically impossible beats, then smooth.
    hr = hr[(hr > 30) & (hr < 220)]
    if hr.size >= 5:
        hr = np.convolve(hr, np.ones(5) / 5, mode="same")
    return peaks, hr


def window_hr(ppg_window, fs):
    """Single HR estimate (bpm) for one analysis window, or NaN if unreliable."""
    _, hr = hr_from_ppg(ppg_window, fs)
    if hr.size == 0:
        return np.nan
    return float(np.median(hr))      # median = robust to a stray bad interval


def majority_activity(activity_window):
    """Most common activity ID in a window (the window's label)."""
    if activity_window.size == 0:
        return -1
    return int(np.bincount(activity_window).argmax())


def score_subject(subject):
    """Run the pipeline over one subject on the ground-truth window grid.

    Returns ``(times, est, truth, act_ids)`` aligned per HR window:
      times   - window-centre time (s)
      est     - estimated HR (bpm), NaN where no reliable estimate
      truth   - ground-truth HR (bpm)
      act_ids - majority activity ID in the window
    """
    bvp = subject["bvp"]
    truth = subject["hr_true"]
    activity = subject["activity"]

    win = int(DALIA_HR_WIN_S * DALIA_FS_BVP)         # 8 s of BVP samples
    act_win = int(DALIA_HR_WIN_S * DALIA_FS_ACT)     # 8 s of activity samples
    shift_bvp = int(DALIA_HR_SHIFT_S * DALIA_FS_BVP) # 2 s hop, BVP samples
    shift_act = int(DALIA_HR_SHIFT_S * DALIA_FS_ACT) # 2 s hop, activity samples

    times, est, act_ids = [], [], []
    for k in range(truth.size):
        b0 = k * shift_bvp
        if b0 + win > bvp.size:                      # ran past the recording
            truth = truth[:k]
            break
        a0 = k * shift_act
        est.append(window_hr(bvp[b0:b0 + win], DALIA_FS_BVP))
        act_ids.append(majority_activity(activity[a0:a0 + act_win]))
        times.append((b0 + win / 2) / DALIA_FS_BVP)

    return np.array(times), np.array(est), truth, np.array(act_ids)


def accumulate_mae(errors_by_activity, est, truth, act_ids):
    """Add this subject's absolute errors into a {activity_id: [abs_errs]} dict."""
    abs_err = np.abs(est - truth)
    valid = ~np.isnan(abs_err)
    for aid in np.unique(act_ids[valid]):
        errors_by_activity.setdefault(aid, []).extend(
            abs_err[valid & (act_ids == aid)].tolist())


def print_mae_table(errors_by_activity):
    """Print the activity-stratified MAE table (the deliverable)."""
    print(f"\n{'activity':<14}{'MAE (bpm)':>11}{'windows':>10}")
    print("-" * 35)
    all_errs = []
    for aid in sorted(errors_by_activity):
        errs = errors_by_activity[aid]
        all_errs.extend(errs)
        name = DALIA_ACTIVITIES.get(aid, f"id{aid}")
        print(f"{name:<14}{np.mean(errs):>11.2f}{len(errs):>10}")
    print("-" * 35)
    print(f"{'overall':<14}{np.mean(all_errs):>11.2f}{len(all_errs):>10}")


def make_figure(demo, errors_by_activity):
    """Two panels: one subject's HR timeline + MAE-by-activity bar chart."""
    times, est, truth, act_ids = demo
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))

    ax[0].plot(times, truth, color="C0", lw=1.2, alpha=0.6, label="ground truth (ECG)")
    ax[0].plot(times, est, color="C3", lw=1.2, alpha=0.6, label="PPG estimate")
    ax[0].set_title("Heart rate over time — one subject")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("HR (bpm)")
    ax[0].legend(loc="upper right")

    aids = sorted(errors_by_activity)
    names = [DALIA_ACTIVITIES.get(a, f"id{a}") for a in aids]
    maes = [np.mean(errors_by_activity[a]) for a in aids]
    ax[1].bar(names, maes, color="C3")
    ax[1].set_title("Mean absolute error by activity (all subjects)")
    ax[1].set_ylabel("MAE (bpm)")
    ax[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Exercise 2: HR-from-PPG on PPG-DaLiA", fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, "exercise2_hr_pipeline.png")


def main():
    parser = argparse.ArgumentParser(description="HR-from-PPG on PPG-DaLiA.")
    parser.add_argument("--data-dir",
                        default=os.environ.get("PPG_DALIA_DIR", "PPG_FieldStudy"),
                        help="PPG_FieldStudy folder (S1/…/S15). "
                             "Defaults to $PPG_DALIA_DIR or ./PPG_FieldStudy")
    parser.add_argument("--subject", default=None,
                        help="Process only one subject, e.g. S3 (default: all)")
    args = parser.parse_args()

    paths = find_dalia_subjects(args.data_dir)
    if args.subject:
        paths = [p for p in paths if os.path.basename(p) == f"{args.subject}.pkl"]
    if not paths:
        sys.exit(
            f"No PPG-DaLiA subjects found under '{args.data_dir}'.\n"
            "Download 'PPG-DaLiA' from the UCI ML Repository (~3 GB), unzip it,\n"
            "and pass --data-dir /path/to/PPG_FieldStudy (or set $PPG_DALIA_DIR)."
        )

    errors_by_activity = {}
    demo = None
    for path in paths:
        subject = load_dalia_subject(path)
        result = score_subject(subject)
        accumulate_mae(errors_by_activity, *result[1:])
        if demo is None:
            demo = result
        print(f"Processed {subject['subject']:>4}  "
              f"({result[0].size} windows)")

    print_mae_table(errors_by_activity)
    path = make_figure(demo, errors_by_activity)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
