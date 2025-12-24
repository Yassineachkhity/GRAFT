"""
Failure-aware reward shaping and critic distillation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from graft.graphs import GraphState
from graft.utils import sigmoid


@dataclass
class FailureLabels:
    """Failure labels for a single step."""

    progress: float
    flags: Dict[str, float]


class FailureTaxonomy:
    """Container for failure modes used by the judge and critic."""

    def __init__(self, modes: Sequence[str]):
        self.modes = list(modes)
        self.index = {mode: idx for idx, mode in enumerate(self.modes)}


class FailureJudge:
    """Base failure judge interface."""

    def label_step(
        self,
        info: Dict,
        graph_state: GraphState,
    ) -> FailureLabels:
        raise NotImplementedError


class HeuristicFailureJudge(FailureJudge):
    """
    Toy heuristic judge that uses environment-provided info signals.
    """

    def __init__(self, taxonomy: FailureTaxonomy):
        self.taxonomy = taxonomy

    def label_step(
        self,
        info: Dict,
        graph_state: GraphState,
    ) -> FailureLabels:
        flags = {mode: 0.0 for mode in self.taxonomy.modes}
        if info.get("invalid_actions"):
            flags["invalid_dependency"] = 1.0
        if info.get("repeated_actions"):
            flags["repeat_action"] = 1.0
        if info.get("idle_actions"):
            flags["idle_with_work"] = 1.0
        if info.get("ignored_messages"):
            flags["ignored_message"] = 1.0
        progress = float(info.get("progress", graph_state.completion_ratio()))
        return FailureLabels(progress=progress, flags=flags)


class ProcessRewardShaper:
    """Combines progress and failure penalties into a process reward."""

    def __init__(
        self,
        progress_weight: float,
        failure_weights: Dict[str, float],
        taxonomy: FailureTaxonomy,
    ):
        self.progress_weight = progress_weight
        self.failure_weights = dict(failure_weights)
        self.taxonomy = taxonomy

    def compute(self, labels: FailureLabels) -> float:
        reward = self.progress_weight * labels.progress
        for mode in self.taxonomy.modes:
            reward -= self.failure_weights.get(mode, 0.0) * float(labels.flags.get(mode, 0.0))
        return reward


class DefaultFeatureExtractor:
    """
    Basic feature extractor for the toy environment.
    """

    def __init__(self, num_tasks: int, num_agents: int):
        self.num_tasks = num_tasks
        self.num_agents = num_agents
        self.action_size = num_tasks + 1
        self.dim = 2 * num_tasks + num_agents + 1

    def extract(self, obs_by_agent: Dict, info: Dict, graph_state: GraphState) -> np.ndarray:
        completed_mask, available_mask = graph_state.masks()
        last_actions = [
            obs_by_agent[a].last_action for a in sorted(obs_by_agent.keys())
        ]
        action_scaled = [a / max(self.action_size - 1, 1) for a in last_actions]
        step_index = float(info.get("step_index", 0))
        max_steps = float(info.get("max_steps", 1))
        step_norm = step_index / max(max_steps, 1.0)
        features = completed_mask + available_mask + action_scaled + [step_norm]
        return np.asarray(features, dtype=float)


class DistilledCritic:
    """
    Lightweight linear critic that predicts progress and failure probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        taxonomy: FailureTaxonomy,
        lr: float = 0.05,
        l2: float = 0.0,
        seed: int = 7,
    ):
        rng = np.random.default_rng(seed)
        self.taxonomy = taxonomy
        self.lr = lr
        self.l2 = l2
        self.W_fail = rng.normal(0.0, 0.01, (len(taxonomy.modes), input_dim))
        self.b_fail = np.zeros(len(taxonomy.modes), dtype=float)
        self.W_prog = rng.normal(0.0, 0.01, (input_dim,))
        self.b_prog = 0.0

    def predict(self, features: np.ndarray) -> FailureLabels:
        logits = self.W_fail @ features + self.b_fail
        probs = np.array([sigmoid(x) for x in logits])
        progress = float(np.clip(self.W_prog @ features + self.b_prog, 0.0, 1.0))
        flags = {mode: float(probs[idx]) for mode, idx in self.taxonomy.index.items()}
        return FailureLabels(progress=progress, flags=flags)

    def train_batch(self, batch: List[Tuple[np.ndarray, FailureLabels]]) -> float:
        if not batch:
            return 0.0
        total_loss = 0.0
        for features, labels in batch:
            preds = self.predict(features)
            # Progress MSE
            prog_error = preds.progress - labels.progress
            total_loss += prog_error * prog_error
            grad_prog = 2.0 * prog_error * features
            self.W_prog -= self.lr * (grad_prog + self.l2 * self.W_prog)
            self.b_prog -= self.lr * 2.0 * prog_error
            # Failure BCE
            for mode, idx in self.taxonomy.index.items():
                target = float(labels.flags.get(mode, 0.0))
                pred = float(preds.flags.get(mode, 0.0))
                total_loss += -(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))
                grad = (pred - target) * features
                self.W_fail[idx] -= self.lr * (grad + self.l2 * self.W_fail[idx])
                self.b_fail[idx] -= self.lr * (pred - target)
        return total_loss / len(batch)


class DistillationBuffer:
    """Simple FIFO buffer for distillation data."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._data: List[Tuple[np.ndarray, FailureLabels]] = []

    def add(self, features: np.ndarray, labels: FailureLabels) -> None:
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append((features, labels))

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, FailureLabels]]:
        if not self._data:
            return []
        batch_size = min(batch_size, len(self._data))
        indices = np.random.choice(len(self._data), size=batch_size, replace=False)
        return [self._data[idx] for idx in indices]
