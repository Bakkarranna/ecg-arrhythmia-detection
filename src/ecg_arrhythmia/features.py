"""RR-interval features: the input to the Random Forest baseline."""

from __future__ import annotations

import numpy as np

from . import config


def rr_series(r_peaks: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """RR intervals in seconds; rrs[i] spans r_peaks[i] to r_peaks[i + 1]."""
    return np.diff(r_peaks) / fs


def beat_rr_features(r_peaks: np.ndarray) -> np.ndarray:
    """Per-beat features (n_beats, N_RR_FEATURES), all in seconds:

    [pre_rr, post_rr, pre_pre_rr, post_post_rr, local_avg_rr]

    pre_rr is the interval ending at this beat, post_rr the one leaving it.
    Vectorized over the whole record.
    """
    rrs = np.diff(r_peaks).astype(np.float64) / config.FS
    if len(rrs) == 0:  # a lone R peak still needs one all-zero feature row
        return np.zeros((max(len(r_peaks), 1), config.N_RR_FEATURES), dtype=np.float32)
    pre = np.concatenate([[rrs[0]], rrs])           # pre[i] = rrs[i - 1]
    post = np.concatenate([rrs, [rrs[-1]]])          # post[i] = rrs[i]
    pre_pre = np.concatenate([pre[:2], pre[:-2]])     # pre_pre[i] = pre[i - 2]
    post_post = np.concatenate([post[2:], post[-2:]])  # post_post[i] = post[i + 2]
    feats = np.column_stack([pre, post, pre_pre, post_post, (pre + post) / 2.0])
    return feats.astype(np.float32)
