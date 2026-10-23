# ECG Arrhythmia Detection

Heartbeat-level arrhythmia classification on the **PhysioNet MIT-BIH Arrhythmia
Database**, using an **honest inter-patient split** and two models compared on
the same test patients: a classical Random-Forest baseline over RR-interval
features, and a 1D CNN over the raw beat waveform plus those RR features.

The point of the project is methodological: report metrics that cannot be
inflated by leakage, on patients the models have never seen.

---

## Headline result

| Model | Macro recall | Macro precision | Macro F1 |
| --- | --- | --- | --- |
| RandomForest (RR intervals) | 0.4490 | 0.4186 | 0.4269 |
| **BeatCNN (1D CNN + RR)** | **0.5101** | 0.4502 | **0.4588** |

Evaluated on **51,149 held-out DS2 beats** from 22 patients never used in
training. Full per-class numbers: [results/cnn_metrics.json](results/cnn_metrics.json),
[results/baseline_metrics.json](results/baseline_metrics.json).

The CNN gains ~3.2 macro-F1 points over the RR baseline. Both numbers are low
in absolute terms, and that is expected and reported honestly: MIT-BIH's five
AAMI classes are dominated by `N` (normal), the arrhythmic classes are tiny and
noisy, and macro-F1 weights all five classes equally. `Q` (unknown/paced-ectopic
leftovers) is the main drag on both models.

---

## The dataset

- **Source:** PhysioNet MIT-BIH Arrhythmia Database — 48 half-hour ECG records,
  47 patients, sampled at 360 Hz, two channels (usually MLII, modified lead II).
- **Fetched at runtime** with `wfdb` directly from PhysioNet; no manual download,
  no bundled binaries.
- **Excluded records:** 102, 104, 107, 217 (paced), following de Chazal et al.,
  because pacing morphology is out of scope for a beat-classifier.
- **Classes (AAMI five-way):** `N` normal, `S` supraventricular ectopic,
  `V` ventricular ectopic, `F` fusion, `Q` unknown. Every annotation symbol is
  mapped through a table and anything unmapped lands in `Q` rather than being
  silently dropped.

## Honest evaluation — why this matters more than the score

1. **Inter-patient split (DS1 / DS2).** DS1 (records 101, 106, 108, 109, 112,
   114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
   223, 230) is train-only; DS2 is test-only. **No patient appears on both
   sides.** Random beat-level splits leak patient-identity signal and inflate
   scores by a wide margin — that is the single most common failure mode in
   published MIT-BIH results.
2. **No beat-level shuffling across the split.** Beats are extracted once, then
   partitioned by record, not by beat index.
3. **Metrics:** per-class precision/recall/F1, macro averages, and a full
   confusion matrix. Bare accuracy is never reported — on MIT-BIH accuracy is
   ~88% from predicting `N` for everything.
4. **Class weighting** in both models, because `N` is ~88% of beats and an
   unweighted classifier degenerates to predicting the majority class.

## Pipeline

```
PhysioNet (MIT-BIH)            raw records + beat annotations
        |  download.py  (wfdb, record whitelist)
        v
filter + resample check        preprocess.py  (band-pass / baseline handling)
        |  data.py  (map_aami, R-peak windows: -0.25 s / +0.45 s => 252 samples)
        v
features.py  (RR intervals, prev/next RR, local RR stats)
        |
        +--> baseline.py   RandomForest on RR features
        |
        +--> models.py     1D CNN (BeatCNN) over the 252-sample waveform,
               train.py     concatenated with the same RR features,
                             weighted cross-entropy, early stop on val macro-F1
        v
evaluate.py       per-class + macro metrics, confusion matrix
visualize.py      figures
results_table.py  cross-model comparison table (CSV + Markdown)
```

**Beat windows** are cut around each annotated R peak — 0.25 s before, 0.45 s
after — giving 252 samples at 360 Hz. Windows that overhang the record edges are
dropped, and the kept beat indices are carried through so per-beat RR features
stay aligned with the waveform.

**Training:** weighted cross-entropy, Adam, batch 256, early stopping on DS1
validation macro-F1. The CNN ran 12 epochs with validation early-stopping,
~1,289 s on CPU. Best validation macro-F1 was 0.3592; test macro-F1 0.4588.
(The test set is larger and easier; the held-out validation split is drawn from
the same 22 DS1 patients as training, which is also why both numbers are
modest.)

## Figures

All in [results/figures/](results/figures):

| Figure | What it shows |
| --- | --- |
| `class_distribution.png` | The ~88% `N` imbalance that motivates class weighting |
| `ecg_strip_100.png` | Raw record 100 strip with annotated beats (sanity check on the data) |
| `mean_beat_waveforms.png` | Per-class mean beat shape — the morphology signal the CNN uses |
| `rr_intervals.png` | RR-interval distributions per class — the signal the baseline uses |
| `training_curves.png` | Loss and macro-F1 across epochs |
| `confusion_baseline.png` | Baseline confusion matrix on DS2 |
| `confusion_cnn.png` | CNN confusion matrix on DS2 |

## Layout

```
src/ecg_arrhythmia/
  config.py        paths, constants, AAMI class list, DS1/DS2 record lists
  download.py      PhysioNet fetch via wfdb (excludes paced records)
  preprocess.py    signal conditioning
  data.py          annotation -> AAMI mapping, beat-window extraction
  features.py      RR-interval features
  models.py        BeatCNN (1D CNN + RR head)
  baseline.py      RandomForest baseline
  train.py         training loop, seeding, standardisation, early stopping
  evaluate.py      metrics + confusion matrix
  visualize.py     all figures
  results_table.py cross-model comparison
tests/             unit tests (pure numpy/scipy, no torch, no network)
.github/workflows/ ci.yml  (test job + torch import-smoke job)
requirements.txt   full deps; requirements-smoke.txt for the test suite alone
notes.md           chronological decisions, data bugs, honesty rules
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# fetch data from PhysioNet (network required)
python -m ecg_arrhythmia.download

# baseline
python -m ecg_arrhythmia.baseline
# CNN (CPU-runnable, ~20 min)
python -m ecg_arrhythmia.train
# figures + comparison table
python -m ecg_arrhythmia.visualize
python -m ecg_arrhythmia.results_table
```

Tests (no network, no torch, ~15 s):

```bash
pip install -r requirements-smoke.txt
pytest -q
ruff check src tests
```

## Tests and CI

`tests/test_ecg_pipeline.py` covers the parts that actually break: AAMI symbol
mapping including the unknown-symbol fallback, beat-window boundary dropping and
index alignment, RR-feature construction order, signal filtering shape/DC
removal, split integrity (no shared records between DS1/DS2), metric
computation on a hand-checked confusion matrix, and reproducibility of seeding.

CI runs two jobs: a `test` job (pytest + ruff, minimal deps) and a `smoke` job
(CPU torch, imports every module to catch API drift).

## Known limitations

- Macro-F1 is ~0.46; this is a legitimate, reproducible result on the hard
  five-way task, not a tuned state-of-the-art number.
- `F` (fusion) and `V` are severely under-represented; per-class recall for them
  is poor and visible in the confusion matrices.
- Only lead MLII is used; the second channel is ignored.
- The model is a research artefact, **not a medical device** and not validated
  for clinical use.

## Credits / references

- Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database*, IEEE Eng in
  Med & Biol, 2001.
- de Chazal, O'Dwyer & Reilly, *Automatic classification of heartbeats using
  ECG morphology and heartbeat interval features*, IEEE TBME, 2004 (source of
  the DS1/DS2 split and the paced-record exclusion list used here).
- ANSI/AAMI EC57 standard for the five-way beat-class nomenclature.
