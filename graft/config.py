"""
Configuration dataclasses for GRAFT components.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PlannerConfig:
    """Planner settings."""

    ensemble_size: int = 4
    diversity_temperature: float = 0.8
    max_nodes: int = 12


@dataclass
class BanditConfig:
    """Exp3 bandit settings for graph selection."""

    gamma: float = 0.2
    min_prob: float = 0.02


@dataclass
class FailureConfig:
    """Failure-aware reward shaping and distillation settings."""

    progress_weight: float = 1.0
    failure_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "repeat_action": 0.2,
            "invalid_dependency": 0.4,
            "idle_with_work": 0.2,
            "ignored_message": 0.2,
        }
    )
    use_judge: bool = True
    judge_warmup_episodes: int = 10
    critic_lr: float = 0.05
    critic_l2: float = 0.0


@dataclass
class CommConfig:
    """Communication gating and cost settings."""

    gate_threshold: float = 0.5
    comm_cost_weight: float = 0.01


@dataclass
class TrainConfig:
    """Training loop settings."""

    episodes: int = 200
    max_steps: int = 50
    seed: int = 7
    return_clip: float = 1.0
    verbose: bool = True
