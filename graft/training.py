"""
GRAFT training loop orchestration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from graft.bandit import Exp3Bandit
from graft.communication import GraphAlignedCommunicator
from graft.config import CommConfig, FailureConfig, TrainConfig
from graft.failure import (
    DefaultFeatureExtractor,
    DistillationBuffer,
    DistilledCritic,
    FailureJudge,
    ProcessRewardShaper,
)
from graft.graphs import GraphState
from graft.marl import MARLAlgorithm
from graft.planner import Planner
from graft.types import Observation, TaskSpec
from graft.utils import set_seed


@dataclass
class EpisodeMetrics:
    episode: int
    total_return: float
    normalized_return: float


class GRAFTTrainer:
    """Coordinates GRAFT components in a single training loop."""

    def __init__(
        self,
        env,
        planner: Planner,
        bandit: Exp3Bandit,
        algorithm: MARLAlgorithm,
        communicator: GraphAlignedCommunicator,
        failure_judge: FailureJudge,
        reward_shaper: ProcessRewardShaper,
        feature_extractor: DefaultFeatureExtractor,
        distill_buffer: Optional[DistillationBuffer],
        critic: Optional[DistilledCritic],
        train_config: TrainConfig,
        failure_config: FailureConfig,
        comm_config: CommConfig,
    ):
        self.env = env
        self.planner = planner
        self.bandit = bandit
        self.algorithm = algorithm
        self.communicator = communicator
        self.failure_judge = failure_judge
        self.reward_shaper = reward_shaper
        self.feature_extractor = feature_extractor
        self.distill_buffer = distill_buffer
        self.critic = critic
        self.train_config = train_config
        self.failure_config = failure_config
        self.comm_config = comm_config
        set_seed(train_config.seed)

    def run(self) -> List[EpisodeMetrics]:
        metrics: List[EpisodeMetrics] = []
        for episode in range(self.train_config.episodes):
            task_spec: TaskSpec = self.env.get_task_spec()
            graphs = self.planner.plan(task_spec)
            self.bandit.update_num_arms(len(graphs))
            graph_index = self.bandit.sample()
            active_graph = graphs[graph_index]
            graph_state = GraphState(active_graph)
            obs_by_agent = self.env.reset()
            total_return = 0.0

            for step in range(self.train_config.max_steps):
                graph_state_before = graph_state.copy()
                messages_by_agent, comm_cost = self.communicator.compute_messages(
                    obs_by_agent, graph_state
                )
                actions_by_agent = self.algorithm.select_actions(
                    obs_by_agent, messages_by_agent, graph_state
                )
                step_result = self.env.step(actions_by_agent)
                for subtask_id in step_result.info.get("completed", []):
                    graph_state.mark_completed(subtask_id)

                features = self.feature_extractor.extract(
                    obs_by_agent, step_result.info, graph_state
                )
                use_judge = self.failure_config.use_judge and (
                    episode < self.failure_config.judge_warmup_episodes
                )

                if use_judge or self.critic is None:
                    labels = self.failure_judge.label_step(step_result.info, graph_state)
                    if self.distill_buffer is not None:
                        self.distill_buffer.add(features, labels)
                else:
                    labels = self.critic.predict(features)

                proc_reward = self.reward_shaper.compute(labels)
                shaped_reward = (
                    step_result.reward
                    + proc_reward
                    - self.comm_config.comm_cost_weight * comm_cost
                )
                total_return += shaped_reward

                self.algorithm.observe_transition(
                    obs_by_agent=obs_by_agent,
                    actions_by_agent=actions_by_agent,
                    reward=shaped_reward,
                    next_obs_by_agent=step_result.obs_by_agent,
                    done=step_result.done,
                    graph_state=graph_state_before,
                    next_graph_state=graph_state,
                )

                obs_by_agent = step_result.obs_by_agent
                if step_result.done:
                    break

            self.algorithm.update()
            if self.critic is not None and self.distill_buffer is not None:
                batch = self.distill_buffer.sample(batch_size=32)
                self.critic.train_batch(batch)

            normalized = self._normalize_return(total_return)
            self.bandit.update(graph_index, normalized)
            metrics.append(
                EpisodeMetrics(
                    episode=episode,
                    total_return=total_return,
                    normalized_return=normalized,
                )
            )
            if self.train_config.verbose and (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{self.train_config.episodes} "
                    f"return={total_return:.2f} normalized={normalized:.2f}"
                )
        return metrics

    def _normalize_return(self, total_return: float) -> float:
        clip = max(self.train_config.return_clip, 1e-6)
        normalized = total_return / clip
        return float(max(0.0, min(1.0, normalized)))
