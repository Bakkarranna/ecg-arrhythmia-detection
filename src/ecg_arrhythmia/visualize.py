"""Figures for the README: class mix, per-class beats, confusion matrices,
training curves and an annotated real ECG strip.

Usage: python -m ecg_arrhythmia.visualize
Writes results/figures/*.png (matplotlib 'Agg' backend, headless-safe).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config, data  # noqa: E402

CLASS_COLORS = {"N": "#2b8a3e", "S": "#1971c2", "V": "#c92a2a", "F": "#f08c00", "Q": "#862e9c"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    })


def fig_class_distribution(fig_dir: Path, npz_path: Path) -> Path:
    z = np.load(npz_path)
    labels, records = z["labels"], z["records"]
    classes = config.AAMI_CLASSES
    ds1 = [(labels == c)[np.isin(records, config.DS1)].sum() for c in classes]
    ds2 = [(labels == c)[np.isin(records, config.DS2)].sum() for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, ds1, 0.4, label="DS1 (train/val)", color="#1971c2")
    ax.bar(x + 0.2, ds2, 0.4, label="DS2 (test)", color="#f08c00")
    for i, (a, b) in enumerate(zip(ds1, ds2)):
        ax.text(i - 0.2, a, f"{a:,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(classes, CLASS_NAMES)])
    ax.set_ylabel("beats")
    ax.set_title("AAMI class distribution across the inter-patient split")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


CLASS_NAMES = {
    "N": "normal", "S": "supraventricular", "V": "ventricular",
    "F": "fusion", "Q": "unknown",
}


def fig_mean_beats(fig_dir: Path, npz_path: Path) -> Path:
    z = np.load(npz_path)
    waves, labels = z["waves"], z["labels"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(config.BEAT_LEN) / config.FS
    t -= config.PRE_SAMPLES / config.FS
    for c in config.AAMI_CLASSES:
        sel = labels == c
        if not sel.any():
            continue
        mean = waves[sel].mean(axis=0)
        std = waves[sel].std(axis=0)
        ax.plot(t, mean, color=CLASS_COLORS[c], lw=1.8, label=f"{c} ({CLASS_NAMES[c]})")
        ax.fill_between(t, mean - std, mean + std, color=CLASS_COLORS[c], alpha=0.12)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("time relative to R peak (s)")
    ax.set_ylabel("amplitude (mV, normalized)")
    ax.set_title("Mean filtered beat waveform per AAMI class (shaded: 1 sd)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = fig_dir / "mean_beat_waveforms.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_confusion(fig_dir: Path, metrics: dict, title: str, fname: str) -> Path:
    cm = np.array(metrics["confusion_matrix"]["matrix"])
    labels = metrics["confusion_matrix"]["labels"]
    cm_n = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nmacro F1 = {metrics['macro_f1']:.3f}")
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_n[i, j] * 100:.0f}%" if cm_n[i, j] >= 0.005 else "-",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_n[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="recall (row-normalized)")
    fig.tight_layout()
    out = fig_dir / fname
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_training_curves(fig_dir: Path, history: dict) -> Path:
    epochs = history["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, history["train_loss"], marker="o", ms=3, color="#c92a2a")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss (weighted CE)")
    ax1.set_title("Training loss")
    ax2.plot(epochs, history["val_macro_f1"], marker="o", ms=3, color="#2b8a3e")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro F1")
    ax2.set_title("Validation macro F1 (held-out DS1 records)")
    fig.tight_layout()
    out = fig_dir / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ecg_strip(fig_dir: Path, record_id: str = "119") -> Path:
    """Real annotated strip: raw signal with true/predicted beat classes."""
    pred_path = config.RESULTS_DIR / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError("predictions.npz missing; run training first")
    preds = np.load(pred_path)
    records = preds["records"]
    if record_id not in set(records.tolist()):
        record_id = str(records[0])
    signal, r_peaks, symbols = data.load_record(record_id)
    filtered = data.bandpass(signal)
    labels = data.map_aami(symbols)
    _, labels, keep = data.extract_beats(filtered, r_peaks, labels)
    peaks = r_peaks[keep]
    in_rec = preds["records"] == record_id
    y_true = preds["y_true"][in_rec]
    y_pred = preds["y_pred"][in_rec]

    start, stop = peaks[5], peaks[min(len(peaks) - 1, 40)]
    stop = min(stop, len(filtered))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(np.arange(start, stop), filtered[start:stop, 0], color="#343a40", lw=0.8)
    for j, p in enumerate(peaks):
        if not (start <= p < stop):
            continue
        k = int(np.flatnonzero(peaks == p)[0])
        cls_t, cls_p = str(y_true[k]), str(y_pred[k])
        ok = cls_t == cls_p
        ax.axvline(p, color=CLASS_COLORS.get(cls_t, "grey"), lw=1.4,
                   alpha=0.9, ls="-" if ok else ":")
        ax.text(p, ax.get_ylim()[1], cls_t, color=CLASS_COLORS.get(cls_t, "grey"),
                ha="center", va="top", fontsize=8)
        if not ok:
            ax.annotate(f"→{cls_p}", (p, ax.get_ylim()[0]), color="#c92a2a",
                        fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("sample (360 Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"MIT-BIH record {record_id}: filtered strip, beat class per R peak\n"
                 f"(solid = correct prediction, dotted red = model disagreement)")
    fig.tight_layout()
    out = fig_dir / f"ecg_strip_{record_id}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_rr_scatter(fig_dir: Path, npz_path: Path) -> Path:
    z = np.load(npz_path)
    rr, labels, records = z["rr"], z["labels"], z["records"]
    mask = np.isin(records, config.DS2) & np.isin(labels, ["N", "V", "S"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in ["N", "V", "S"]:
        sel = mask & (labels == c)
        if sel.sum() > 4000:
            sel_idx = np.flatnonzero(sel)
            sel_idx = np.random.default_rng(config.SEED).choice(sel_idx, 4000, replace=False)
            sel = np.zeros(len(labels), bool)
            sel[sel_idx] = True
        ax.scatter(rr[sel, 0], rr[sel, 1], s=4, alpha=0.35,
                   color=CLASS_COLORS[c], label=f"{c} ({CLASS_NAMES[c]})")
    ax.set_xlabel("pre-RR interval (s)")
    ax.set_ylabel("post-RR interval (s)")
    ax.set_title("RR intervals alone separate many classes (DS2)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    out = fig_dir / "rr_intervals.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(npz_path: Path = None, results_dir: Path = config.RESULTS_DIR) -> list[Path]:
    npz_path = npz_path or config.PROCESSED_DIR / "beats.npz"
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    made = []
    made.append(fig_class_distribution(fig_dir, npz_path))
    made.append(fig_mean_beats(fig_dir, npz_path))
    made.append(fig_rr_scatter(fig_dir, npz_path))
    history_path = Path(results_dir) / "train_history.json"
    if history_path.exists():
        made.append(fig_training_curves(fig_dir, json.loads(history_path.read_text())))
    for fname, title in [("baseline", "Random Forest baseline (RR intervals)"),
                         ("cnn", "BeatCNN")]:
        mpath = Path(results_dir) / f"{fname}_metrics.json"
        if mpath.exists():
            metrics = json.loads(mpath.read_text())
            made.append(fig_confusion(fig_dir, metrics, title, f"confusion_{fname}.png"))
    if (Path(results_dir) / "predictions.npz").exists():
        made.append(fig_ecg_strip(fig_dir))
    print("Figures written to", fig_dir)
    for p in made:
        print(" -", p.name)
    return made


def main() -> None:
    make_all()


if __name__ == "__main__":
    main()
