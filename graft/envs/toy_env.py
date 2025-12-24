"""
Toy multi-agent environment for GRAFT demonstrations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graft.graphs import GraphState, SubtaskGraph, SubtaskNode
from graft.types import Observation, SubtaskSpec, TaskSpec
from graft.utils import set_seed


def build_demo_task() -> TaskSpec:
    """Construct a small, deterministic task for the demo."""

    subtasks = [
        SubtaskSpec("t1", "Pick items", assigned_agent=0),
        SubtaskSpec("t2", "Pack items", assigned_agent=1),
        SubtaskSpec("t3", "Label package", assigned_agent=1),
        SubtaskSpec("t4", "Ship order", assigned_agent=2),
    ]
    dependencies = [("t1", "t2"), ("t2", "t3"), ("t3", "t4")]
    return TaskSpec(
        task_id="demo-1",
        description="Fulfill a warehouse order",
        subtasks=subtasks,
        dependencies=dependencies,
        agent_count=3,
    )


@dataclass
class StepResult:
    obs_by_agent: Dict[int, Observation]
    reward: float
    done: bool
    info: Dict


class ToyMultiAgentEnv:
    """Simple dependency-constrained multi-agent task environment."""

    def __init__(self, task_spec: TaskSpec, max_steps: int = 50, seed: int = 7):
        self.task_spec = task_spec
        self.max_steps = max_steps
        self.rng_seed = seed
        set_seed(seed)
        self.subtask_ids = [s.subtask_id for s in task_spec.subtasks]
        self.subtask_index = {sid: idx for idx, sid in enumerate(self.subtask_ids)}
        self.graph = SubtaskGraph(
            nodes=[
                SubtaskNode(
                    subtask_id=s.subtask_id,
                    description=s.description,
                    assigned_agent=s.assigned_agent,
                )
                for s in task_spec.subtasks
            ],
            edges=task_spec.dependencies,
        )
        self.graph_state = GraphState(self.graph)
        self.step_index = 0
        self.last_actions: Dict[int, int] = {i: 0 for i in range(task_spec.agent_count)}

    def reset(self) -> Dict[int, Observation]:
        self.graph_state = GraphState(self.graph)
        self.step_index = 0
        self.last_actions = {i: 0 for i in range(self.task_spec.agent_count)}
        return self._build_obs()

    def get_task_spec(self) -> TaskSpec:
        return self.task_spec

    @property
    def action_size(self) -> int:
        return len(self.subtask_ids) + 1

    def step(self, actions_by_agent: Dict[int, int]) -> StepResult:
        completed_this_step: List[str] = []
        invalid_actions: List[int] = []
        repeated_actions: List[int] = []
        idle_actions: List[int] = []

        for agent_id, action in actions_by_agent.items():
            if action == 0:
                if self._has_available_task(agent_id):
                    idle_actions.append(agent_id)
                continue
            subtask_id = self._action_to_subtask(action)
            if subtask_id is None:
                invalid_actions.append(agent_id)
                continue
            if self.graph_state.is_completed(subtask_id):
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            if not self._deps_satisfied(subtask_id):
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            assigned = self.graph.nodes[subtask_id].assigned_agent
            if assigned is not None and assigned != agent_id:
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            completed_this_step.append(subtask_id)
            if self.last_actions.get(agent_id) == action:
                repeated_actions.append(agent_id)

        for subtask_id in completed_this_step:
            self.graph_state.mark_completed(subtask_id)

        self.step_index += 1
        done = self.graph_state.completion_ratio() >= 1.0 or self.step_index >= self.max_steps
        reward = float(len(completed_this_step))
        if done and self.graph_state.completion_ratio() >= 1.0:
            reward += 5.0
        reward -= 0.1 * len(invalid_actions)

        self.last_actions.update(actions_by_agent)
        info = {
            "completed": completed_this_step,
            "invalid_actions": invalid_actions,
            "repeated_actions": repeated_actions,
            "idle_actions": idle_actions,
            "progress": self.graph_state.completion_ratio(),
            "step_index": self.step_index,
            "max_steps": self.max_steps,
        }
        obs_by_agent = self._build_obs()
        return StepResult(obs_by_agent=obs_by_agent, reward=reward, done=done, info=info)

    def _action_to_subtask(self, action: int) -> Optional[str]:
        idx = action - 1
        if idx < 0 or idx >= len(self.subtask_ids):
            return None
        return self.subtask_ids[idx]

    def _deps_satisfied(self, subtask_id: str) -> bool:
        preds = self.graph.predecessors(subtask_id)
        return preds.issubset(self.graph_state.completed)

    def _has_available_task(self, agent_id: int) -> bool:
        for sid in self.subtask_ids:
            if self.graph_state.is_completed(sid):
                continue
            if not self._deps_satisfied(sid):
                continue
            assigned = self.graph.nodes[sid].assigned_agent
            if assigned is None or assigned == agent_id:
                return True
        return False

    def _build_obs(self) -> Dict[int, Observation]:
        obs_by_agent = {}
        for agent_id in range(self.task_spec.agent_count):
            completed_mask, _ = self.graph_state.masks()
            available_mask = []
            for sid in self.subtask_ids:
                if self.graph_state.is_completed(sid):
                    available_mask.append(0)
                    continue
                if not self._deps_satisfied(sid):
                    available_mask.append(0)
                    continue
                assigned = self.graph.nodes[sid].assigned_agent
                if assigned is None or assigned == agent_id:
                    available_mask.append(1)
                else:
                    available_mask.append(0)
            obs_by_agent[agent_id] = Observation(
                agent_id=agent_id,
                completed_mask=list(completed_mask),
                available_mask=available_mask,
                last_action=self.last_actions.get(agent_id, 0),
                step_index=self.step_index,
            )
        return obs_by_agent
