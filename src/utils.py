from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    computes cosine similarity between two 1D numpy arrays.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError("Zero-norm embedding encountered.")

    return float(np.dot(a, b) / (a_norm * b_norm))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    moves a tensor to CPU and convert to a numpy array.
    """
    return tensor.detach().cpu().numpy()