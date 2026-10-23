"""Preprocess CLI: download MIT-BIH records and build the processed dataset.

Usage:  python -m ecg_arrhythmia.preprocess [--records 100 101 ...]
Writes data/processed/beats.npz with waves, labels, rr features and record ids.
RR features are computed per record then sliced to the beats that survived
extraction, so rows stay aligned with the waveforms.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config, data, download, features


def all_record_ids() -> list[str]:
    return sorted(set(config.DS1 + config.DS2))


def load_processed(npz_path: Path = config.PROCESSED_DIR / "beats.npz") -> dict:
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def preprocess(record_ids: list[str], raw_dir: Path = config.RAW_DIR,
               out_path: Path = config.PROCESSED_DIR / "beats.npz") -> Path:
    waves_all, labels_all, rr_all, rec_all = [], [], [], []
    bad = download.download_records(record_ids, raw_dir)
    record_ids = [r for r in record_ids if r not in bad]
    for rid in record_ids:
        t0 = time.time()
        signal, r_peaks, symbols = data.load_record(rid, raw_dir)
        filtered = data.bandpass(signal)
        labels = data.map_aami(symbols)
        waves, labels, keep = data.extract_beats(filtered, r_peaks, labels)
        rr = features.beat_rr_features(r_peaks)[keep]
        waves_all.append(waves)
        labels_all.append(labels)
        rr_all.append(rr)
        rec_all.append(np.full(len(waves), rid, dtype=object))
        print(f"  {rid}: {len(waves):6d} beats  ({time.time() - t0:.1f}s)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        waves=np.concatenate(waves_all).astype(np.float32),
        labels=np.concatenate(labels_all).astype(str),
        rr=np.concatenate(rr_all).astype(np.float32),
        records=np.concatenate(rec_all).astype(str),
    )
    print(f"Saved {out_path} ({len(np.concatenate(labels_all))} beats)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="*", default=None,
                        help="Record ids (default: the 44 split records)")
    args = parser.parse_args()
    ids = args.records or all_record_ids()
    print(f"Preprocessing {len(ids)} records...")
    preprocess(ids)


if __name__ == "__main__":
    main()
