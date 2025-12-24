"""
Subtask graph utilities for GRAFT.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class SubtaskNode:
    """A node representing a subtask in the plan graph."""

    subtask_id: str
    description: str
    assigned_agent: Optional[int] = None
    completion_criteria: Optional[str] = None


class SubtaskGraph:
    """Directed acyclic graph (DAG) of subtasks and dependencies."""

    def __init__(self, nodes: Sequence[SubtaskNode], edges: Sequence[Tuple[str, str]]):
        self.nodes: Dict[str, SubtaskNode] = {n.subtask_id: n for n in nodes}
        self.edges: List[Tuple[str, str]] = list(edges)
        self._pred: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        self._succ: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Edge references unknown node: ({u}, {v})")
            self._pred[v].add(u)
            self._succ[u].add(v)
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        """Raise ValueError if a cycle is detected."""

        indeg = {n: len(self._pred[n]) for n in self.nodes}
        queue = [n for n, d in indeg.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for nxt in self._succ[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if visited != len(self.nodes):
            raise ValueError("SubtaskGraph contains a cycle")

    def predecessors(self, node_id: str) -> Set[str]:
        return set(self._pred[node_id])

    def successors(self, node_id: str) -> Set[str]:
        return set(self._succ[node_id])

    def ready_nodes(self, completed: Set[str]) -> List[str]:
        """Return nodes that are not completed and whose predecessors are complete."""

        ready = []
        for node_id in self.nodes:
            if node_id in completed:
                continue
            if self._pred[node_id].issubset(completed):
                ready.append(node_id)
        return ready

    def node_order(self) -> List[str]:
        """Stable node order for masks and feature extraction."""

        return list(self.nodes.keys())

    def agent_edges(self) -> List[Tuple[int, int]]:
        """Map subtask dependencies to agent-to-agent edges."""

        edges: List[Tuple[int, int]] = []
        for u, v in self.edges:
            src = self.nodes[u].assigned_agent
            dst = self.nodes[v].assigned_agent
            if src is None or dst is None:
                continue
            if src == dst:
                continue
            edges.append((src, dst))
        return edges


class GraphState:
    """Tracks execution state for a subtask graph during an episode."""

    def __init__(self, graph: SubtaskGraph):
        self.graph = graph
        self.completed: Set[str] = set()

    def mark_completed(self, subtask_id: str) -> None:
        if subtask_id in self.graph.nodes:
            self.completed.add(subtask_id)

    def is_completed(self, subtask_id: str) -> bool:
        return subtask_id in self.completed

    def ready_nodes(self) -> List[str]:
        return self.graph.ready_nodes(self.completed)

    def completion_ratio(self) -> float:
        if not self.graph.nodes:
            return 1.0
        return len(self.completed) / float(len(self.graph.nodes))

    def copy(self) -> "GraphState":
        """Create a shallow copy of the graph state."""

        cloned = GraphState(self.graph)
        cloned.completed = set(self.completed)
        return cloned

    def masks(self) -> Tuple[List[int], List[int]]:
        """Return completed and available masks in graph node order."""

        order = self.graph.node_order()
        completed_mask = [1 if n in self.completed else 0 for n in order]
        available = set(self.ready_nodes())
        available_mask = [1 if n in available else 0 for n in order]
        return completed_mask, available_mask
