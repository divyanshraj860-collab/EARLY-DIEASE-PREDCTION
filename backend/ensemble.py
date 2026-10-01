"""Lightweight numeric helpers shared by training and serving (no plotting deps)."""
from __future__ import annotations

import numpy as np


def full_proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with columns aligned to label-encoder classes 0..n_classes-1."""
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    out = np.zeros((p.shape[0], n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = p
    return out


def soft_vote(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Soft voting: weighted average of the models' CLASS PROBABILITY matrices.

    Equal weights give  (p_RF + p_NB + p_KNN + p_SVC + p_DT + p_GB) / 6.
    Weights must sum to 1, so the result is again a probability distribution.
    """
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"ensemble weights must sum to 1 (got {total:.4f})")
    return sum(weights[k] * probas[k] for k in weights)
