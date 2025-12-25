"""
Compare single-agent vs multi-agent performance on coding tasks using GRAFT.
"""

import csv
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from graft.bandit import Exp3Bandit
from graft.communication import GateController, GraphAlignedCommunicator, SimpleMessageEncoder
from graft.config import CommConfig, FailureConfig, TrainConfig
from graft.envs.code_env import CodeTaskEnv, build_code_task_suite
from graft.failure import (
    DefaultFeatureExtractor,
    DistillationBuffer,
    DistilledCritic,
    FailureTaxonomy,
    HeuristicFailureJudge,
    ProcessRewardShaper,
)
from graft.marl import IndependentQLearner, SimpleStateEncoder
from graft.planner import GeminiLLMClient, LocalLLMPlanner
from graft.training import EpisodeMetrics, GRAFTTrainer


def load_gemini_config() -> tuple[str, str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_version = os.getenv("GEMINI_API_VERSION", "v1")
    return api_key, model, api_version


def write_metrics(path: Path, metrics: List[EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "total_return",
                "normalized_return",
                "steps",
                "completion_ratio",
                "comm_cost",
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m.episode,
                    f"{m.total_return:.4f}",
                    f"{m.normalized_return:.4f}",
                    m.steps,
                    f"{m.completion_ratio:.4f}",
                    f"{m.comm_cost:.4f}",
                ]
            )


def moving_average(series: List[float], window: int = 5) -> List[float]:
    if window <= 1:
        return series
    values = np.asarray(series, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    return smoothed.tolist()


def run_config(label: str, agent_assignments: List[int], agent_count: int) -> List[EpisodeMetrics]:
    task_spec, tasks, dependencies = build_code_task_suite(agent_assignments, agent_count)
    env = CodeTaskEnv(task_spec=task_spec, tasks=tasks, dependencies=dependencies, max_steps=40, seed=7)

    api_key, model, api_version = load_gemini_config()
    client = GeminiLLMClient(api_key=api_key, model=model, api_version=api_version)
    planner = LocalLLMPlanner(client=client, ensemble_size=3)
    bandit = Exp3Bandit(num_arms=planner.ensemble_size, gamma=0.2, seed=7)

    failure_config = FailureConfig()
    taxonomy = FailureTaxonomy(list(failure_config.failure_weights.keys()))
    judge = HeuristicFailureJudge(taxonomy)
    reward_shaper = ProcessRewardShaper(
        progress_weight=failure_config.progress_weight,
        failure_weights=failure_config.failure_weights,
        taxonomy=taxonomy,
    )

    feature_extractor = DefaultFeatureExtractor(
        num_tasks=len(task_spec.subtasks),
        num_agents=task_spec.agent_count,
    )
    critic = DistilledCritic(
        input_dim=feature_extractor.dim,
        taxonomy=taxonomy,
        lr=failure_config.critic_lr,
        l2=failure_config.critic_l2,
    )
    distill_buffer = DistillationBuffer(capacity=2000)

    comm_config = CommConfig()
    communicator = GraphAlignedCommunicator(
        encoder=SimpleMessageEncoder(),
        gate=GateController(threshold=comm_config.gate_threshold),
    )

    algorithm = IndependentQLearner(
        num_agents=task_spec.agent_count,
        action_size=env.action_size,
        encoder=SimpleStateEncoder(),
        lr=0.2,
        gamma=0.95,
        epsilon=0.1,
    )

    trainer = GRAFTTrainer(
        env=env,
        planner=planner,
        bandit=bandit,
        algorithm=algorithm,
        communicator=communicator,
        failure_judge=judge,
        reward_shaper=reward_shaper,
        feature_extractor=feature_extractor,
        distill_buffer=distill_buffer,
        critic=critic,
        train_config=TrainConfig(episodes=60, max_steps=40, return_clip=8.0, verbose=True),
        failure_config=failure_config,
        comm_config=comm_config,
    )
    metrics = trainer.run()
    print(f"{label} final completion ratio: {metrics[-1].completion_ratio:.2f}")
    return metrics


def plot_comparison(
    single_metrics: List[EpisodeMetrics],
    multi_metrics: List[EpisodeMetrics],
    output_path: Path,
) -> None:
    episodes = [m.episode for m in single_metrics]
    single_completion = [m.completion_ratio for m in single_metrics]
    multi_completion = [m.completion_ratio for m in multi_metrics]
    single_return = [m.normalized_return for m in single_metrics]
    multi_return = [m.normalized_return for m in multi_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(episodes, moving_average(single_completion), label="single-agent")
    axes[0].plot(episodes, moving_average(multi_completion), label="multi-agent")
    axes[0].set_title("Completion Ratio")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Completion")
    axes[0].legend()

    axes[1].plot(episodes, moving_average(single_return), label="single-agent")
    axes[1].plot(episodes, moving_average(multi_return), label="multi-agent")
    axes[1].set_title("Normalized Return")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    single_metrics = run_config("single-agent", agent_assignments=[0, 0, 0], agent_count=1)
    multi_metrics = run_config("multi-agent", agent_assignments=[0, 1, 2], agent_count=3)

    write_metrics(runs_dir / "single_agent_metrics.csv", single_metrics)
    write_metrics(runs_dir / "multi_agent_metrics.csv", multi_metrics)
    plot_comparison(single_metrics, multi_metrics, runs_dir / "comparison.png")


if __name__ == "__main__":
    main()
