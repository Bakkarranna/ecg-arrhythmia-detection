"""Tests for the ECG arrhythmia pipeline (no network, no data files)."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_arrhythmia import config, data, evaluate, features
from ecg_arrhythmia.models import BeatCNN


def test_rr_features_are_vectorized_and_correct():
    r_peaks = np.array([0, 100, 200, 500, 900])
    feats = features.beat_rr_features(r_peaks)
    rrs = np.diff(r_peaks) / config.FS
    assert feats.shape == (5, config.N_RR_FEATURES)
    np.testing.assert_allclose(feats[:, 0], np.concatenate([[rrs[0]], rrs]))       # pre
    np.testing.assert_allclose(feats[:, 1], np.concatenate([rrs, [rrs[-1]]]))      # post
    np.testing.assert_allclose(feats[:, 2], np.concatenate([feats[:2, 0], feats[:-2, 0]]))
    np.testing.assert_allclose(feats[:, 3], np.concatenate([feats[2:, 1], feats[-2:, 1]]))
    np.testing.assert_allclose(feats[:, 4], (feats[:, 0] + feats[:, 1]) / 2)


def test_rr_features_handle_single_beat():
    feats = features.beat_rr_features(np.array([100]))
    assert feats.shape == (1, config.N_RR_FEATURES)
    assert np.all(feats == 0)


def test_aami_mapping_covers_every_standard_class():
    symbols = np.array(["N", "V", "S", "F", "Q", "L", "j", "e"])
    mapped = data.map_aami(symbols)
    assert set(mapped.tolist()) <= set(config.AAMI_CLASSES)
    assert mapped[1] == "V" and mapped[3] == "F"


def test_extract_beats_drops_overhang_and_keeps_alignment():
    n = 400
    fs = config.FS
    t = np.arange(n) / fs
    signal = np.stack([np.sin(2 * np.pi * 1.2 * t)] * 2, axis=1)  # (n, 2)
    r_peaks = np.array([config.PRE_SAMPLES + 5, 200, n - config.PRE_SAMPLES - 5])
    labels = np.array(["N", "V", "S"], dtype=object)
    waves, kept_labels, keep = data.extract_beats(signal, r_peaks, labels)
    assert waves.shape == (2, config.BEAT_LEN)
    assert keep.tolist() == [0, 1]          # last two peaks overhang
    assert kept_labels.tolist() == ["N", "V"]
    np.testing.assert_allclose(waves[1], signal[200 - config.PRE_SAMPLES:200 + config.POST_SAMPLES, 0])


def test_bandpass_preserves_shape_and_removes_dc():
    fs = config.FS
    t = np.arange(fs * 5) / fs
    signal = 5.0 + np.sin(2 * np.pi * 1.0 * t)
    out = data.bandpass(signal[:, None])
    assert out.shape == signal[:, None].shape
    assert abs(np.mean(out)) < 0.05        # baseline wander dropped
    assert np.ptp(out) > 1.0               # 1 Hz content survives


def test_beatcnn_forward_shapes():
    torch = pytest.importorskip("torch")
    model = BeatCNN()
    waves = np.random.randn(8, config.BEAT_LEN).astype(np.float32)
    rr = np.random.randn(8, config.N_RR_FEATURES).astype(np.float32)

    out = model(torch.as_tensor(waves), torch.as_tensor(rr))
    assert out.shape == (8, len(config.AAMI_CLASSES))
    assert torch.isfinite(out).all()


def test_metrics_on_tiny_example():
    y_true = np.array(["N", "S", "V", "Q", "F", "N"])
    y_pred = np.array(["N", "S", "V", "N", "N", "S"])
    m = evaluate.compute_metrics(y_true, y_pred)
    assert m["n_samples"] == 6
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert set(m["per_class"]) == set(config.AAMI_CLASSES)
    cm = np.array(m["confusion_matrix"]["matrix"])
    assert cm.sum() == 6
    assert m["class_distribution"]["N"] == 2


def test_interpatient_split_is_disjoint():
    assert not set(config.DS1) & set(config.DS2)
    assert not (set(config.TRAIN_RECORDS) & set(config.VAL_RECORDS))
    assert set(config.TRAIN_RECORDS) | set(config.VAL_RECORDS) == set(config.DS1)
    for p in config.PACEMAKER_RECORDS:
        assert p not in config.DS1 and p not in config.DS2
