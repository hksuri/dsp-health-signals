"""
Step 1 — dissect ONE afdb record, field by field.

Before windowing or plotting anything, it pays to open a single record and look
at every byte you're given. An afdb record is THREE files that share a name:

    04015.hea   — the header: plain text, describes the signal
    04015.dat   — the signal: packed binary, the actual ECG samples
    04015.atr   — rhythm annotations: when each rhythm starts (the labels)
    04015.qrs   — beat annotations: where each heartbeat is (machine-detected)

This script prints what's inside each one and explains every field, so when a
later script says ``ann.aux_note`` or ``record.p_signal`` you know exactly what
that is and where it came from. Nothing here is saved — it's a read-out.

Run:  python anatomy_of_a_record.py                 # default 04015
      python anatomy_of_a_record.py --record 08219
"""

import argparse
import os

import numpy as np
import wfdb

from utils import AFDB_DIR, _afdb, RHYTHM_NAMES


def show_header(rec):
    """The .hea file — plain text. Print it raw, then decode wfdb's view."""
    name, kw = _afdb(rec)
    print("=" * 64)
    print(f".hea  —  the header (plain text)")
    print("=" * 64)
    hea_path = os.path.join(AFDB_DIR, rec + ".hea")
    if os.path.exists(hea_path):
        print("Raw file:")
        for line in open(hea_path):
            print("    " + line.rstrip())
    hdr = wfdb.rdheader(name, **kw)
    print("\nDecoded:")
    print(f"    record name      : {hdr.record_name}")
    print(f"    # signals        : {hdr.n_sig}  (the two ECG leads)")
    print(f"    sampling rate fs : {hdr.fs} Hz")
    print(f"    # samples/signal : {hdr.sig_len:,}  "
          f"(= {hdr.sig_len/hdr.fs/3600:.1f} h)")
    if hdr.n_sig:
        print(f"    channel names    : {hdr.sig_name}")
        print(f"    units            : {hdr.units}")
        print(f"    ADC gain         : {hdr.adc_gain}  (ADC units per mV)")
        print(f"    storage format   : {hdr.fmt}  (212 = 12-bit packed)")
    else:
        print("    (no signal — this is one of the annotation-only records)")
    return hdr


def show_signal(rec, hdr, seconds=2):
    """The .dat file — the samples themselves. Read a couple of seconds."""
    print("\n" + "=" * 64)
    print(".dat  —  the signal (packed binary → millivolts)")
    print("=" * 64)
    if hdr.sig_len == 0:
        print("    no .dat for this record; skipping.")
        return
    name, kw = _afdb(rec, "dat")
    n = seconds * hdr.fs
    r = wfdb.rdrecord(name, sampfrom=0, sampto=n, **kw)
    sig = r.p_signal                                  # shape (n, 2), in mV
    print(f"    p_signal shape   : {sig.shape}  (samples × channels)")
    print(f"    dtype            : {sig.dtype}  (already scaled to mV via gain)")
    print(f"    first 5 samples of {r.sig_name[0]} (mV): "
          f"{np.round(sig[:5, 0], 3).tolist()}")
    print(f"    {r.sig_name[0]} range over {seconds}s: "
          f"[{sig[:,0].min():.2f}, {sig[:,0].max():.2f}] mV")
    print("    → raw .dat stores integers; wfdb divides by ADC gain to give mV.")


def show_rhythm_annotations(rec):
    """The .atr file — the labels. Each marks where a rhythm STARTS."""
    print("\n" + "=" * 64)
    print(".atr  —  rhythm annotations (the ground-truth labels)")
    print("=" * 64)
    name, kw = _afdb(rec, "atr")
    ann = wfdb.rdann(name, "atr", **kw)
    print(f"    # annotations    : {len(ann.sample)}")
    print(f"    fields per annotation: sample, symbol, aux_note, subtype, chan, num")
    print("    For afdb, rhythm lives in aux_note; symbol is '+' (a rhythm change).\n")
    print(f"    {'#':>3}  {'sample':>9}  {'time(s)':>8}  {'sym':>3}  "
          f"{'aux_note':<8}  → rhythm")
    print("    " + "-" * 52)
    for i in range(min(8, len(ann.sample))):
        aux = (ann.aux_note[i] or "").strip().strip("\x00")
        rhythm = RHYTHM_NAMES.get(aux, aux)
        print(f"    {i:>3}  {ann.sample[i]:>9}  {ann.sample[i]/ann.fs:>8.1f}  "
              f"{ann.symbol[i]:>3}  {aux:<8}  → {rhythm}")
    print("\n    Each row says 'at this sample, rhythm switches to X, and stays")
    print("    X until the next row.' That's why utils.rhythm_spans() pairs each")
    print("    start with the NEXT start to rebuild (start, end, rhythm) spans.")


def show_beat_annotations(rec):
    """The .qrs file — beat fiducials. Source of RR intervals."""
    print("\n" + "=" * 64)
    print(".qrs  —  beat annotations (machine-detected QRS, source of RR)")
    print("=" * 64)
    name, kw = _afdb(rec, "qrs")
    ann = wfdb.rdann(name, "qrs", **kw)
    print(f"    # beats          : {len(ann.sample):,}")
    print(f"    first 8 beat samples : {ann.sample[:8].tolist()}")
    rr = np.diff(ann.sample[:9]) / ann.fs
    print(f"    → RR intervals (s)   : {np.round(rr, 3).tolist()}")
    print("    These are NOT hand-checked beats — they're an automatic detector's")
    print("    output, so RR-based features inherit some detection error.")


def main():
    p = argparse.ArgumentParser(description="Dissect one afdb record's files.")
    p.add_argument("--record", default="04015", help="afdb record id")
    args = p.parse_args()
    rec = args.record

    print(f"\n### Anatomy of afdb record {rec} ###\n")
    hdr = show_header(rec)
    show_signal(rec, hdr)
    show_rhythm_annotations(rec)
    show_beat_annotations(rec)
    print("\nThat's the whole record: header + signal + two annotation streams.")


if __name__ == "__main__":
    main()
