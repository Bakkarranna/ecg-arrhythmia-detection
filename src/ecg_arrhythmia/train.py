"""Train the 1D CNN beat classifier with a patient-level validation split.

Honest protocol:
  * DS1 records (minus VAL_RECORDS) -> training
  * VAL_RECORDS (held-out block of DS1) -> model selection only
  * DS2 records (never seen) -> final reported test metrics

Usage: python -m ecg_arrhythmia.train [--epochs 12] [--batch-size 128]
Writes results/beat_cnn.pt, results/cnn_metrics.json, results/train_history.json
and results/predictions.npz (for the figures in visualize.py).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, evaluate, models, preprocess


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_rr(
    rr_train: np.ndarray, rr_rest: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a scaler on training RR features only, apply it to the rest.

    Returns (scaled_train, scaled_rest, mean, std)."""
    mean = rr_train.mean(axis=0)
    std = rr_train.std(axis=0) + 1e-6
    return (rr_train - mean) / std, (rr_rest - mean) / std, mean, std


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = {c: int((labels == c).sum()) for c in config.AAMI_CLASSES}
    k = len(config.AAMI_CLASSES)
    w = torch.tensor([len(labels) / (k * max(counts[c], 1)) for c in config.AAMI_CLASSES],
                     dtype=torch.float32, device=device)
    return w


def loader_for(waves, rr, labels, batch: int, shuffle: bool) -> DataLoader:
    xs = torch.as_tensor(waves, dtype=torch.float32)
    rs = torch.as_tensor(rr, dtype=torch.float32)
    ys = torch.as_tensor(labels_to_idx(labels), dtype=torch.long)
    return DataLoader(TensorDataset(xs, rs, ys), batch_size=batch, shuffle=shuffle)


def labels_to_idx(labels: np.ndarray) -> np.ndarray:
    table = {c: i for i, c in enumerate(config.AAMI_CLASSES)}
    return np.array([table[x] for x in labels], dtype=np.int64)


def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer,
          device: torch.device) -> float:
    model.train()
    total, n = 0.0, 0
    for waves, rr, y in loader:
        waves, rr, y = waves.to(device), rr.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(waves, rr)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    for waves, rr, y in loader:
        out = model(waves.to(device), rr.to(device))
        preds.append(out.argmax(dim=1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_training(npz_path: Path = None, epochs: int = None, batch_size: int = None,
                 out_dir: Path = None, verbose: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = npz_path or config.PROCESSED_DIR / "beats.npz"
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()

    bundle = preprocess.load_processed(npz_path)
    waves, rr, labels, records = bundle["waves"], bundle["rr"], bundle["labels"], bundle["records"]

    train_m = np.isin(records, config.TRAIN_RECORDS)
    val_m = np.isin(records, config.VAL_RECORDS)
    test_m = np.isin(records, config.DS2)

    # Scaler fitted on training beats only, then applied to val and test.
    mean = rr[train_m].mean(axis=0)
    std = rr[train_m].std(axis=0) + 1e-6
    rr_all = (rr - mean) / std

    # Per-beat amplitude normalization: each window is zero-mean/unit-variance,
    # which removes the electrode-gain differences between records.
    waves_all = (waves - waves.mean(axis=1, keepdims=True)) / (
        waves.std(axis=1, keepdims=True) + 1e-6)

    y_train = labels[train_m]
    train_loader = loader_for(waves_all[train_m], rr_all[train_m], labels[train_m], batch_size, shuffle=True)
    val_loader = loader_for(waves_all[val_m], rr_all[val_m], labels[val_m], batch_size, shuffle=False)
    test_loader = loader_for(waves_all[test_m], rr_all[test_m], labels[test_m], batch_size, shuffle=False)

    model = models.BeatCNN().to(device)
    w = class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epochs": [], "val_macro_f1": [], "train_loss": []}
    best_f1, best_state = -1.0, None
    patience, bad_epochs = 4, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        y_true, y_pred = predict(model, val_loader, device)
        val_f1 = float(f1_score(y_true, y_pred, average="macro"))
        history["epochs"].append(epoch)
        history["train_loss"].append(loss)
        history["val_macro_f1"].append(val_f1)
        if verbose:
            print(f"epoch {epoch:2d}/{epochs}  loss {loss:.4f}  val macro F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(best_state)
    scaler_mean = rr[train_m].mean(axis=0)
    scaler_std = rr[train_m].std(axis=0) + 1e-6
    ckpt = {
        "state_dict": model.state_dict(),
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "classes": config.AAMI_CLASSES,
        "best_val_macro_f1": best_f1,
    }
    torch.save(ckpt, out_dir / config.MODEL_PATH.name)
    y_true_idx, y_pred_idx = predict(model, test_loader, device)
    y_true_s = np.array(config.AAMI_CLASSES)[y_true_idx]
    y_pred_s = np.array(config.AAMI_CLASSES)[y_pred_idx]
    metrics = evaluate.compute_metrics(y_true_s, y_pred_s)
    metrics["model"] = "BeatCNN(1D CNN + RR features)"
    metrics["best_val_macro_f1"] = round(best_f1, 4)
    metrics["epochs_ran"] = len(history["epochs"])
    metrics["training_seconds"] = round(time.time() - t0, 1)
    evaluate.save_metrics(metrics, out_dir / "cnn_metrics.json")
    np.savez_compressed(out_dir / "predictions.npz",
                        y_true=y_true_s, y_pred=y_pred_s, records=records[test_m],
                        wave_idx=np.flatnonzero(test_m))
    history["best_val_macro_f1"] = best_f1
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    if verbose:
        print(f"Test macro F1: {metrics['macro_f1']:.4f}  (best val {best_f1:.4f})")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run_training(Path(args.npz) if args.npz else None, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
