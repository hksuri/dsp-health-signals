# dsp-health-signals — working notes

A learning portfolio of classic DSP for wearable health signals. Two topic
folders, each with self-contained `exerciseN_*.py` scripts that save figures
into their own `figures/`:

- `filters-and-detection/` — band-pass, Pan-Tompkins QRS, notch, spectrogram + NLMS.
- `ppg/` — AC/DC + perfusion index, HR-from-PPG (PPG-DaLiA), HRV, (SpO₂ planned).

## Python environment

Use the conda env **`ct-view`** — it has every dependency these scripts need
(numpy, scipy, matplotlib, **wfdb**). There is no project-local `.venv`.

```bash
# interpreter (use directly, no activation needed):
/Users/huzaifasuri/opt/anaconda3/envs/ct-view/bin/python <script>.py

# or activate it:
conda activate ct-view
```

`wfdb` is required by the exercises that pull real PhysioNet data
(`filters-and-detection/exercise2_pan_tompkins.py`, `ppg/exercise3_hrv.py` —
both download MIT-BIH record 100). The system/base `python3` does **not** have
`wfdb`, so prefer the `ct-view` interpreter when running anything here.

## External datasets (not committed)

- **MIT-BIH** (PhysioNet) — downloaded on demand via `wfdb`, not redistributed.
- **PPG-DaLiA** (~19 GB unpacked) — lives outside the repo. Point scripts at it
  with `--data-dir` or `export PPG_DALIA_DIR=/path/to/PPG_FieldStudy`.

## Git / pushing

The git credential helper is `store` with no cached GitHub credential, so plain
`git push` hangs on an interactive username prompt in non-interactive shells.
Until `gh auth setup-git` is run once, push via the gh token in the URL:

```bash
TOKEN=$(gh auth token)
git push "https://x-access-token:${TOKEN}@github.com/hksuri/dsp-health-signals.git" main:main
```
