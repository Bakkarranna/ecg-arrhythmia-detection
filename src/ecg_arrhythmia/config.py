"""Configuration, constants and hyperparameters for the ECG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "mitdb"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "beat_cnn.pt"

FS = 360  # MIT-BIH sampling frequency (Hz)
PRE_SAMPLES = 90  # 0.25 s before the R peak
POST_SAMPLES = 162  # 0.45 s after the R peak
BEAT_LEN = PRE_SAMPLES + POST_SAMPLES

# Band-pass filter (0.5-40 Hz) removing baseline wander and high-frequency noise.
FILTER_ORDER = 3
FILTER_LOW = 0.5
FILTER_HIGH = 40.0

# Annotation symbols grouped into the five AAMI recommendation classes.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# Records from patients with pacemakers are excluded, per the AAMI standard.
PACEMAKER_RECORDS = ["102", "104", "107", "217"]

# Inter-patient split from de Chazal et al. (2006): no patient in both sets.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

# Held-out block of DS1 used only for model selection (early stopping,
# best-checkpoint). Keeps the reported DS2 numbers honest.
VAL_RECORDS = ["106", "109", "112", "118", "203", "207", "209", "223"]
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]

SEED = 42
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3


@dataclass(frozen=True)
class BeatSet:
    """Beat windows plus labels and provenance."""

    waves: object  # np.ndarray (n_beats, BEAT_LEN)
    labels: object  # np.ndarray (n_beats,) of AAMI class strings
    rr: object  # np.ndarray (n_beats, N_RR_FEATURES)
    records: object  # np.ndarray (n_beats,) of source record ids


N_RR_FEATURES = 5
