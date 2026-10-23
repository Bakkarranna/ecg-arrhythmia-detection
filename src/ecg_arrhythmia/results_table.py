"""Combine every model's metrics into one README-ready results table.

Reads results/*_metrics.json and writes results/results_table.md + .csv, so the
README numbers are generated from the actual runs instead of hand-typed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import config

MODEL_ORDER = ["RandomForest(rr_intervals)", "BeatCNN(1D CNN + RR features)"]
METRIC_KEYS = [
    ("macro_recall", "Macro recall"), ("macro_precision", "Macro precision"),
    ("macro_f1", "Macro F1"),
]


def collect(results_dir: Path = config.RESULTS_DIR) -> dict:
    rows = {}
    for path in sorted(results_dir.glob("*_metrics.json")):
        metrics = json.loads(path.read_text())
        rows[metrics.get("model", path.stem)] = metrics
    return rows


def render(rows: dict) -> tuple[str, list[dict]]:
    lines = [
        "| Model | Macro recall | Macro precision | Macro F1 |",
        "| --- | --- | --- | --- |",
    ]
    table = []
    for model in MODEL_ORDER:
        m = rows.get(model)
        if not m:
            continue
        vals = [f"{m[key]:.4f}" for key, _ in METRIC_KEYS]
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
        table.append({"model": model, **{key: m[key] for key, _ in METRIC_KEYS}})
    return "\n".join(lines) + "\n", table


def write_table(results_dir: Path = config.RESULTS_DIR) -> str:
    rows = collect(results_dir)
    md, table = render(rows)
    (results_dir / "results_table.md").write_text(md)
    with open(results_dir / "results_table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model"] + [k for k, _ in METRIC_KEYS])
        writer.writeheader()
        writer.writerows(table)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()
    print(write_table(Path(args.results_dir)), end="")


if __name__ == "__main__":
    main()
