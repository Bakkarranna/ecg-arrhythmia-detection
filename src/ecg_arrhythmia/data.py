"""Download, filter and beat-extraction for the MIT-BIH Arrhythmia Database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from . import config, download

EXTENSIONS = ("hea", "dat", "atr")


def download_record(record_id: str, dest: Path = config.RAW_DIR) -> Path:
    """Ensure one record's .hea/.dat/.atr exist locally (size-validated)."""
    bad = download.download_records([record_id], dest, workers=EXTENSIONS.__len__())
    if record_id in bad:
        raise RuntimeError(f"could not download record {record_id} completely")
    return dest


def load_record(record_id: str, raw_dir: Path = config.RAW_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signal, r_peak_samples, symbols) for one local record."""
    import wfdb

    record = wfdb.rdrecord(str(raw_dir / record_id))
    ann = wfdb.rdann(str(raw_dir / record_id), "atr")
    signal = record.p_signal.astype(np.float32)
    return signal, np.asarray(ann.sample, dtype=np.int64), np.asarray(ann.symbol)


def bandpass(signal: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """Zero-phase 0.5-40 Hz band-pass removing baseline wander and noise."""
    b, a = butter(config.FILTER_ORDER, [config.FILTER_LOW, config.FILTER_HIGH], btype="band", fs=fs)
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def map_aami(symbols: np.ndarray) -> np.ndarray:
    """Map annotation symbols to the five AAMI classes; unknown -> 'Q'."""
    return np.array([config.AAMI_MAP.get(s, "Q") for s in symbols],dtype=object)


def extract_beats(
    signal: np.ndarray, r_peaks: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut fixed-length windows around each R peak; drop beats that overhang.

    Returns (waves, labels, keep_idx) where keep_idx indexes into r_peaks so
    per-beat features (RR intervals) stay aligned with the kept beats.
    """
    n = len(signal)
    lo, hi = r_peaks - config.PRE_SAMPLES, r_peaks + config.POST_SAMPLES
    keep = np.flatnonzero((lo >= 0) & (hi <= n))
    waves = np.empty((len(keep), config.BEAT_LEN), dtype=np.float32)
    for row, i in enumerate(keep):
        waves[row] = signal[lo[i]:hi[i], 0]
    return waves, np.asarray(labels)[keep], keep


def build_record_beats(record_id: str, raw_dir: Path = config.RAW_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Filtered signal -> AAMI-labeled beat windows for one record."""
    signal, r_peaks, symbols = load_record(record_id, raw_dir)
    filtered = bandpass(signal)
    labels = map_aami(symbols)
    return (*extract_beats(filtered, r_peaks, labels),)
