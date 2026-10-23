"""Baseline: RR intervals + Random Forest (DS1 train, DS2 test)."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config, evaluate


def beatset_from_npz(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def class_weight_dict(y: np.ndarray) -> dict:
    counts = {c: int((y == c).sum()) for c in config.AAMI_CLASSES}
    n, k = len(y), len(config.AAMI_CLASSES)
    return {c: n / (k * max(counts[c], 1)) for c in config.AAMI_CLASSES}


def run_baseline(npz_path=None, out_path=None) -> dict:
    npz_path = npz_path or config.PROCESSED_DIR / "beats.npz"
    out_path = out_path or config.RESULTS_DIR / "baseline_metrics.json"
    data = beatset_from_npz(npz_path)
    rr, labels, records = data["rr"], data["labels"], data["records"]
    train_mask = np.isin(records, config.DS1)
    test_mask = np.isin(records, config.DS2)

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=config.SEED, n_jobs=-1,
    )
    clf.fit(rr[train_mask], labels[train_mask])
    pred = clf.predict(rr[test_mask])

    metrics = evaluate.compute_metrics(labels[test_mask], pred)
    metrics["model"] = "RandomForest(rr_intervals)"
    evaluate.save_metrics(metrics, out_path)
    print(f"Baseline macro F1: {metrics['macro_f1']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_baseline()


if __name__ == "__main__":
    main()
