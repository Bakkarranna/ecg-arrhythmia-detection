import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from . import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class precision/recall/F1 plus macro averages, as a JSON dict."""
    labels = config.AAMI_CLASSES
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {
        "n_samples": int(len(y_true)),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {c: {k: round(float(v), 4) for k, v in report[c].items()} for c in labels},
        "class_distribution": {c: int((np.asarray(y_true) == c).sum()) for c in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    return out


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
