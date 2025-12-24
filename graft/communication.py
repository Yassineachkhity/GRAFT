"""
Graph-aligned communication and gating.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from graft.graphs import GraphState, SubtaskGraph
from graft.utils import sigmoid


@dataclass
class Message:
    """Simple message container."""

    sender: int
    payload: np.ndarray


class MessageEncoder:
    """Base message encoder interface."""

    def encode(self, obs, graph_state: GraphState) -> np.ndarray:
        raise NotImplementedError


class SimpleMessageEncoder(MessageEncoder):
    """Encodes a small summary vector."""

    def encode(self, obs, graph_state: GraphState) -> np.ndarray:
        completed_mask, available_mask = graph_state.masks()
        ratio = graph_state.completion_ratio()
        payload = np.asarray(
            completed_mask + available_mask + [ratio, float(obs.agent_id)],
            dtype=float,
        )
        return payload


class GateController:
    """Per-edge sigmoid gating with a simple learnable weight."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.weights: Dict[Tuple[int, int], float] = {}

    def gate(self, edge: Tuple[int, int], feature: float) -> float:
        weight = self.weights.get(edge, 1.0)
        return sigmoid(weight * feature)

    def should_send(self, edge: Tuple[int, int], feature: float) -> Tuple[float, bool]:
        value = self.gate(edge, feature)
        return value, value >= self.threshold

    def update(self, edge: Tuple[int, int], grad: float, lr: float = 0.01) -> None:
        self.weights[edge] = self.weights.get(edge, 1.0) - lr * grad


class GraphAlignedCommunicator:
    """Restricts and gates messages based on the active subtask graph."""

    def __init__(
        self,
        encoder: MessageEncoder,
        gate: GateController,
    ):
        self.encoder = encoder
        self.gate = gate

    def compute_messages(
        self,
        obs_by_agent: Dict[int, object],
        graph_state: GraphState,
    ) -> Tuple[Dict[int, np.ndarray], float]:
        graph: SubtaskGraph = graph_state.graph
        edges = graph.agent_edges()
        messages_by_agent: Dict[int, List[np.ndarray]] = {
            agent_id: [] for agent_id in obs_by_agent
        }
        comm_cost = 0.0
        for src, dst in edges:
            if src not in obs_by_agent or dst not in obs_by_agent:
                continue
            payload = self.encoder.encode(obs_by_agent[src], graph_state)
            feature = float(graph_state.completion_ratio())
            gate_value, send = self.gate.should_send((src, dst), feature)
            comm_cost += gate_value
            if send:
                messages_by_agent[dst].append(gate_value * payload)
        aggregated = {}
        for agent_id, msgs in messages_by_agent.items():
            if not msgs:
                aggregated[agent_id] = np.zeros(1, dtype=float)
            else:
                aggregated[agent_id] = np.sum(msgs, axis=0)
        return aggregated, comm_cost
