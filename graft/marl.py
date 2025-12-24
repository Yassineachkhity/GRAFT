"""
Minimal MARL algorithm interface and a simple independent Q-learner.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from graft.types import Observation


class StateEncoder:
    """Encodes an observation into a hashable state."""

    def encode(self, obs: Observation, graph_state) -> Tuple[int, ...]:
        raise NotImplementedError


class SimpleStateEncoder(StateEncoder):
    """Concatenates masks and last action into a discrete state."""

    def encode(self, obs: Observation, graph_state) -> Tuple[int, ...]:
        _, available_mask = graph_state.masks()
        return tuple(obs.completed_mask + available_mask + [obs.last_action])


class MARLAlgorithm:
    """Base interface for MARL learners."""

    def select_actions(
        self,
        obs_by_agent: Dict[int, Observation],
        messages_by_agent: Dict[int, np.ndarray],
        graph_state,
    ) -> Dict[int, int]:
        raise NotImplementedError

    def observe_transition(
        self,
        obs_by_agent: Dict[int, Observation],
        actions_by_agent: Dict[int, int],
        reward: float,
        next_obs_by_agent: Dict[int, Observation],
        done: bool,
        graph_state,
        next_graph_state,
    ) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError


@dataclass
class Transition:
    state: Tuple[int, ...]
    action: int
    reward: float
    next_state: Tuple[int, ...]
    done: bool


class IndependentQLearner(MARLAlgorithm):
    """Simple independent Q-learning for the toy demo."""

    def __init__(
        self,
        num_agents: int,
        action_size: int,
        encoder: StateEncoder,
        lr: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.num_agents = num_agents
        self.action_size = action_size
        self.encoder = encoder
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_tables: List[Dict[Tuple[int, ...], np.ndarray]] = [
            {} for _ in range(num_agents)
        ]
        self.buffers: List[List[Transition]] = [[] for _ in range(num_agents)]

    def select_actions(
        self,
        obs_by_agent: Dict[int, Observation],
        messages_by_agent: Dict[int, np.ndarray],
        graph_state,
    ) -> Dict[int, int]:
        actions = {}
        for agent_id, obs in obs_by_agent.items():
            state = self.encoder.encode(obs, graph_state)
            q_values = self._get_q(agent_id, state)
            if np.random.rand() < self.epsilon:
                action = int(np.random.randint(self.action_size))
            else:
                action = int(np.argmax(q_values))
            actions[agent_id] = action
        return actions

    def observe_transition(
        self,
        obs_by_agent: Dict[int, Observation],
        actions_by_agent: Dict[int, int],
        reward: float,
        next_obs_by_agent: Dict[int, Observation],
        done: bool,
        graph_state,
        next_graph_state,
    ) -> None:
        for agent_id in obs_by_agent:
            state = self.encoder.encode(obs_by_agent[agent_id], graph_state)
            next_state = self.encoder.encode(next_obs_by_agent[agent_id], next_graph_state)
            action = int(actions_by_agent[agent_id])
            self.buffers[agent_id].append(
                Transition(state, action, reward, next_state, done)
            )

    def update(self) -> None:
        for agent_id in range(self.num_agents):
            buffer = self.buffers[agent_id]
            for transition in buffer:
                q_values = self._get_q(agent_id, transition.state)
                next_q = self._get_q(agent_id, transition.next_state)
                target = transition.reward
                if not transition.done:
                    target += self.gamma * float(np.max(next_q))
                q_values[transition.action] += self.lr * (target - q_values[transition.action])
            buffer.clear()

    def _get_q(self, agent_id: int, state: Tuple[int, ...]) -> np.ndarray:
        table = self.q_tables[agent_id]
        if state not in table:
            table[state] = np.zeros(self.action_size, dtype=float)
        return table[state]
