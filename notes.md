# ECG Arrhythmia Detection - working notes

Chronological log of decisions, data issues and results. Anything in **bold**
is a decision that shaped the final artefact.

## Data

- **Source: PhysioNet MIT-BIH Arrhythmia Database**, 48 half-hour records,
  47 patients, 360 Hz, dual-channel (MLII usually modified lead II).
- Records are fetched with `wfdb` straight from PhysioNet (no manual download).
- Paced records **102, 104, 107, 217** are excluded, per de Chazal et al.
- Split is the **standard inter-patient split (DS1 / DS2)**, not random.
- Beat windows: 0.25 s before / 0.45 s after each annotated R-peak -> 252
  samples at 360 Hz.

## Corruption incident (fix: `data/raw/mitdb` partial downloads)

Symptom: `wfdb.rdann` raised on some records, and loaded arrays had length 0.

Cause: an interrupted resumable download left truncated `.dat` files with a
valid `.hea`/`.atr`, so wfdb read fewer samples than the header promised.

Fix: re-download the affected records with `wfdb` and verify with
`rdrecord().sig_len == 650000` and a non-empty annotation count.

Verified-good records (spot check): 106, 111, 117, 123 -> each
`sig_len = 650000` and 1500-2100 annotations.

## Model / training

- 1D CNN over the 252-sample window of the MLII channel.
- Class weights are used because N dominates the class distribution.
- Reported metrics: per-class precision / recall, macro F1, confusion matrix,
  on DS2 (patients never seen in training).

## Honesty rules

- No patient appears in both DS1 and DS2.
- No beat-level shuffling between train and test.
- Metric set: macro F1 and per-class recall, never bare accuracy.
