"""
Step 0 — get the data onto disk (and keep it out of git).

Project 1 lives on two PhysioNet datasets. This script pulls both into a local,
git-ignored ``data/`` folder so every other script can read from disk instead of
re-streaming from PhysioNet on each run:

  data/
    afdb/            ← MIT-BIH Atrial Fibrillation Database (the main dataset)
    challenge2017/   ← PhysioNet/CinC Challenge 2017 (single-lead, 4 classes)

Why two datasets? They are two angles on the same question:
  • afdb            — long (~10 h) two-lead ECGs with *cardiologist rhythm
                      labels marking exactly when AFib starts and stops. This is
                      where we learn what AFib looks like and build windows.
  • challenge 2017  — thousands of *short* (~30 s) single-lead strips, each with
                      ONE label (Normal / AF / Other / Noisy). This is the shape
                      a wrist-style classifier actually sees, and a sanity check
                      that our ideas generalise past the 25 afdb patients.

Nothing here is committed — see ../.gitignore (``project1-afib-detection/data/``).

Run:  python download_data.py            # both datasets, skips what's present
      python download_data.py --afdb     # just one of them
      python download_data.py --c2017
"""

import argparse
import os
import urllib.request
import zipfile

import wfdb

from utils import DATA_DIR, AFDB_DIR, C2017_DIR, AFDB_DB

# The Challenge-2017 *training* set ships as one zip (not WFDB-listable), so we
# fetch it directly rather than via wfdb.dl_database.
C2017_ZIP_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"


def _human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def download_afdb():
    """All 25 afdb records (signals + .atr rhythm + .qrs beat annotations)."""
    os.makedirs(AFDB_DIR, exist_ok=True)
    recs = wfdb.get_record_list(AFDB_DB)
    print(f"afdb: {len(recs)} records → {AFDB_DIR}")
    # dl_database is idempotent-ish but re-checks every file; skip if we already
    # have a header for each record.
    have = [r for r in recs if os.path.exists(os.path.join(AFDB_DIR, r + ".hea"))]
    if len(have) == len(recs):
        print(f"  already present ({len(have)}/{len(recs)}), skipping.")
        return
    print(f"  have {len(have)}/{len(recs)}; downloading the rest (~650 MB total)…")
    wfdb.dl_database(AFDB_DB, dl_dir=AFDB_DIR)
    print("  done.")


def download_challenge2017():
    """The Challenge-2017 training set: ~8.5k short single-lead ECGs + labels."""
    os.makedirs(C2017_DIR, exist_ok=True)
    ref = os.path.join(C2017_DIR, "REFERENCE.csv")
    if os.path.exists(ref):
        n = sum(1 for _ in open(ref))
        print(f"challenge2017: already present ({n} labelled records), skipping.")
        return

    zip_path = os.path.join(DATA_DIR, "training2017.zip")
    if not os.path.exists(zip_path):
        print(f"challenge2017: downloading {C2017_ZIP_URL}")
        print("  (~95 MB) …")
        urllib.request.urlretrieve(C2017_ZIP_URL, zip_path)
    print(f"  downloaded {_human(os.path.getsize(zip_path))}; unzipping…")

    # The zip contains a top-level ``training2017/`` folder; flatten it into
    # data/challenge2017/ so paths are predictable.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = member.split("training2017/", 1)[-1]
            if not name:                       # the directory entry itself
                continue
            target = os.path.join(C2017_DIR, name)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    # The label file ships as REFERENCE-v3.csv (the corrected labels); normalise
    # the name so downstream code just reads REFERENCE.csv.
    v3 = os.path.join(C2017_DIR, "REFERENCE-v3.csv")
    if os.path.exists(v3) and not os.path.exists(ref):
        os.replace(v3, ref)
    os.remove(zip_path)
    n = sum(1 for _ in open(ref)) if os.path.exists(ref) else 0
    print(f"  done — {n} labelled records in {C2017_DIR}")


def main():
    p = argparse.ArgumentParser(description="Download Project-1 datasets.")
    p.add_argument("--afdb", action="store_true", help="only the MIT-BIH AFib DB")
    p.add_argument("--c2017", action="store_true", help="only Challenge 2017")
    args = p.parse_args()

    do_afdb = args.afdb or not args.c2017
    do_c2017 = args.c2017 or not args.afdb

    os.makedirs(DATA_DIR, exist_ok=True)
    if do_afdb:
        download_afdb()
    if do_c2017:
        download_challenge2017()
    print("All requested data is in", DATA_DIR)


if __name__ == "__main__":
    main()
