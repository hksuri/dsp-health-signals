# DSP for Wearable Health Signals

> Applied digital signal processing for PPG and ECG — the same toolkit that turns a noisy wrist sensor into a heart-rate number.

![Rest → Walk → Run spectrogram with NLMS motion rejection](figures/exercise4_spectrogram.png)

*Top: a photoplethysmogram (PPG) recorded through a rest → walk → run transition. As intensity climbs, the foot-strike **cadence** band lights up and crowds the **cardiac** band. Middle: after a Normalized-LMS adaptive filter driven by an accelerometer reference, the cadence band is suppressed and the heart-rate line survives. Bottom: the ground-truth cardiac signal for comparison.*

This repository is a compact, hands-on tour of the DSP that lives inside every wearable health device: **filter design, beat detection, denoising, and adaptive motion rejection**. Each technique is implemented from first principles (no black-box detectors), validated on either synthetic signals with known ground truth or real clinical data, and rendered as a figure you can read at a glance. It's built as a learning portfolio — the emphasis is on *why* each method works and *why it matters* for real hardware.

---

## Contents

| # | Script | What it does |
|---|--------|--------------|
| 1 | [`exercise1_bandpass.py`](exercise1_bandpass.py) | Zero-phase Butterworth band-pass that strips baseline wander and HF noise from synthetic PPG and ECG. |
| 2 | [`exercise2_pan_tompkins.py`](exercise2_pan_tompkins.py) | Pan-Tompkins QRS detector built from scratch, validated against **MIT-BIH record 100** (Sensitivity + Positive Predictivity). |
| 3 | [`exercise3_notch.py`](exercise3_notch.py) | IIR notch filter that removes 50/60 Hz powerline hum, verified with a before/after Welch PSD. |
| 4 | [`exercise4_spectrogram_nlms.py`](exercise4_spectrogram_nlms.py) | Spectrogram of a rest→walk→run PPG **+ bonus** NLMS adaptive filter that removes cadence artifact using an accelerometer reference. |

Shared synthetic-signal generators and filter helpers live in [`utils.py`](utils.py) so the algorithms aren't duplicated across scripts.

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/hksuri/dsp-health-signals.git
cd dsp-health-signals

# 2. Install dependencies (a virtualenv is recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run any exercise — each saves its figure(s) into figures/
python exercise1_bandpass.py
python exercise2_pan_tompkins.py   # downloads MIT-BIH record 100 from PhysioNet
python exercise3_notch.py
python exercise4_spectrogram_nlms.py
```

Every script is self-contained, prints a short summary to the console, and writes PNGs into `figures/`. Exercise 2 fetches MIT-BIH record 100 from PhysioNet via the `wfdb` package; if the download (or `wfdb` itself) is unavailable, it prints a notice and still produces the synthetic-data figure.

---

## Walkthrough

### 1 — Butterworth Band-Pass Filtering

![Band-pass on synthetic PPG and ECG](figures/exercise1_bandpass.png)

A band-pass filter keeps only the frequency band where the physiology lives and discards everything else — here, slow **baseline wander** (<0.5 Hz, from breathing and electrode motion) below the band, and **high-frequency noise** above it. The filter is applied with `sosfiltfilt`, a forward-backward pass that is **zero-phase**: it introduces no time delay and so doesn't shift or smear beat timing. That linear-phase property is non-negotiable for ECG, where the relative position of the P, QRS, and T waves *is* the diagnostic information — a filter that delayed different frequencies by different amounts would distort the very morphology a clinician reads.

### 2 — Pan-Tompkins QRS Detection

![Pan-Tompkins on MIT-BIH record 100](figures/exercise2_pan_tompkins_mitbih.png)

Pan-Tompkins is the classic algorithm behind decades of heart-rate monitors. The pipeline shapes the signal so that QRS complexes become unmissable: **band-pass (5–15 Hz)** to isolate QRS energy → **derivative** to emphasize the steep R-wave slope → **squaring** to make everything positive and amplify large slopes → **moving-window integration** to form a smooth energy envelope → an **adaptive threshold** to pick peaks, which are then back-tracked to the true R-peak in the filtered signal. Scored against the cardiologist annotations on **MIT-BIH record 100**, this from-scratch implementation reaches **Sensitivity = 1.000** and **Positive Predictivity = 1.000** (2273 true positives, 1 false positive, 0 misses, ±50 ms tolerance) — the green ✕ annotations and red detections sit right on top of each other. The synthetic-data run (`exercise2_pan_tompkins_synthetic.png`) provides a controlled sanity check with known beat positions.

### 3 — Powerline (50/60 Hz) Notch Filter

![Notch filter PSD before/after](figures/exercise3_notch.png)

Mains electricity radiates a narrow, persistent tone into any biopotential recording — 60 Hz in North America, 50 Hz across much of the rest of the world. A **zero-phase IIR notch** (`iirnotch` + `filtfilt`) carves out a thin slice of spectrum at exactly that frequency while leaving the neighbouring ECG content untouched. The **Welch power spectral density** before and after makes the effect unambiguous: the 60 Hz spike drops by **~48 dB** (from −55.5 dB to −103.3 dB), essentially erasing the hum without blunting the QRS. A notch is preferred over simply low-passing below 60 Hz because real ECG carries useful energy *above* the powerline frequency that you don't want to throw away.

### 4 — Spectrogram + NLMS Adaptive Motion Rejection

![Time-domain NLMS motion rejection](figures/exercise4_timedomain.png)

The spectrogram at the top of this README is the headline result; this time-domain view shows the mechanism. During exercise, the **cadence** (steps per second) drives a large motion artifact into the PPG — and crucially, cadence and heart rate occupy *overlapping* frequencies that both drift upward with intensity, so no fixed filter can separate them. The wearable's trick is an extra sensor: an **accelerometer** that sees the motion but not the blood flow. A **Normalized LMS (NLMS)** adaptive filter learns, sample by sample, the mapping from the accelerometer to the motion component contaminating the PPG, then subtracts it. What's left (the error signal) is the cardiac waveform. NLMS normalizes its step size by the input power, so it stays stable whether the wearer is strolling or sprinting — which is exactly why adaptive reference-based cancellation, not a smarter band-pass, is what production optical heart-rate sensors actually rely on.

---

## Notes

- **Synthetic signals** are used wherever ground truth matters, so each algorithm can be scored against a known answer rather than judged by eye.
- All figures are committed under `figures/` so they render directly on GitHub; re-running any script regenerates them.
- MIT-BIH data is © PhysioNet and is downloaded on demand — it is not redistributed in this repo.

## License

MIT — free to use for learning and reference.
