"""
Exp3 bandit for online graph selection.
"""

from typing import List, Optional

import numpy as np


class Exp3Bandit:
    """Exp3 bandit for adversarial bandit selection."""

    def __init__(self, num_arms: int, gamma: float = 0.2, seed: Optional[int] = None):
        if num_arms <= 0:
            raise ValueError("num_arms must be positive")
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self._init_weights(num_arms)

    def _init_weights(self, num_arms: int) -> None:
        self.num_arms = num_arms
        self.weights = np.ones(self.num_arms, dtype=float)

    def update_num_arms(self, num_arms: int) -> None:
        if num_arms != self.num_arms:
            self._init_weights(num_arms)

    def probabilities(self) -> List[float]:
        """Return the sampling distribution over arms."""

        total = float(self.weights.sum())
        probs = (1.0 - self.gamma) * (self.weights / total) + self.gamma / self.num_arms
        return probs.tolist()

    def sample(self) -> int:
        probs = self.probabilities()
        return int(self.rng.choice(self.num_arms, p=probs))

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update weights using importance-weighted reward."""

        if chosen_arm < 0 or chosen_arm >= self.num_arms:
            raise ValueError("chosen_arm out of range")
        reward = max(0.0, min(1.0, reward))
        probs = self.probabilities()
        p = probs[chosen_arm]
        if p <= 0.0:
            return
        est_reward = reward / p
        growth = np.exp((self.gamma * est_reward) / self.num_arms)
        self.weights[chosen_arm] *= growth
