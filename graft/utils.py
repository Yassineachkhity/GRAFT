"""
Utility helpers.
"""

import math
import random
from typing import Iterable, List

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs."""

    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softmax(xs: Iterable[float]) -> List[float]:
    xs = list(xs)
    if not xs:
        return []
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [e / total for e in exps]
