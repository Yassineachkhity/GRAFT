"""
Minimal runnable demo of GRAFT on the toy environment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from graft.bandit import Exp3Bandit
from graft.communication import GateController, GraphAlignedCommunicator, SimpleMessageEncoder
from graft.config import CommConfig, FailureConfig, TrainConfig
from graft.envs.toy_env import ToyMultiAgentEnv, build_demo_task
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
from graft.training import GRAFTTrainer


def load_api_key() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return api_key


def main() -> None:
    task = build_demo_task()
    env = ToyMultiAgentEnv(task_spec=task, max_steps=40, seed=7)

    api_key = load_api_key()
    client = GeminiLLMClient(api_key=api_key, model="gemini-1.5-flash")
    planner = LocalLLMPlanner(client=client, ensemble_size=4)
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
        num_tasks=len(task.subtasks), num_agents=task.agent_count
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
        num_agents=task.agent_count,
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
        train_config=TrainConfig(episodes=50, max_steps=40, return_clip=10.0, verbose=True),
        failure_config=failure_config,
        comm_config=comm_config,
    )

    trainer.run()


if __name__ == "__main__":
    main()
