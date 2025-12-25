"""
Coding-task environment for GRAFT comparisons.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graft.graphs import GraphState, SubtaskGraph, SubtaskNode
from graft.types import Observation, SubtaskSpec, TaskSpec
from graft.utils import set_seed


@dataclass
class CodeTask:
    task_id: str
    description: str
    function_name: str
    candidates: List[str]
    tests: List[Tuple[Tuple, object]]

    def evaluate(self, candidate_idx: int) -> bool:
        if candidate_idx < 0 or candidate_idx >= len(self.candidates):
            return False
        code = self.candidates[candidate_idx]
        namespace: Dict[str, object] = {}
        try:
            exec(code, namespace)
            func = namespace.get(self.function_name)
            if not callable(func):
                return False
            for args, expected in self.tests:
                if func(*args) != expected:
                    return False
            return True
        except Exception:
            return False


def build_code_task_suite(
    agent_assignments: List[int],
    agent_count: int,
) -> Tuple[TaskSpec, List[CodeTask], List[Tuple[str, str]]]:
    tasks = [
        CodeTask(
            task_id="c1",
            description="Implement add(a, b) that returns a + b.",
            function_name="add",
            candidates=[
                "def add(a, b):\n    return a + b\n",
                "def add(a, b):\n    return a - b\n",
                "def add(a, b):\n    return a * b\n",
            ],
            tests=[((1, 2), 3), ((-1, 5), 4)],
        ),
        CodeTask(
            task_id="c2",
            description="Implement is_even(n) that returns True if n is even.",
            function_name="is_even",
            candidates=[
                "def is_even(n):\n    return n % 2 == 0\n",
                "def is_even(n):\n    return n % 2 == 1\n",
                "def is_even(n):\n    return n == 2\n",
            ],
            tests=[((2,), True), ((3,), False), ((0,), True)],
        ),
        CodeTask(
            task_id="c3",
            description="Implement factorial(n) for non-negative n.",
            function_name="factorial",
            candidates=[
                "def factorial(n):\n    out = 1\n    for i in range(1, n + 1):\n        out *= i\n    return out\n",
                "def factorial(n):\n    return n\n",
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n + factorial(n - 1)\n",
            ],
            tests=[((0,), 1), ((3,), 6), ((5,), 120)],
        ),
    ]
    if len(agent_assignments) != len(tasks):
        raise ValueError("agent_assignments length must match tasks")
    dependencies = [("c1", "c2"), ("c2", "c3")]
    subtasks = []
    for task, agent_id in zip(tasks, agent_assignments):
        subtasks.append(
            SubtaskSpec(
                subtask_id=task.task_id,
                description=task.description,
                assigned_agent=agent_id,
            )
        )
    task_spec = TaskSpec(
        task_id="code-suite",
        description="Solve a small suite of coding tasks.",
        subtasks=subtasks,
        dependencies=dependencies,
        agent_count=agent_count,
    )
    return task_spec, tasks, dependencies


@dataclass
class StepResult:
    obs_by_agent: Dict[int, Observation]
    reward: float
    done: bool
    info: Dict


class CodeTaskEnv:
    """Environment where agents choose candidate solutions for coding tasks."""

    def __init__(
        self,
        task_spec: TaskSpec,
        tasks: List[CodeTask],
        dependencies: List[Tuple[str, str]],
        max_steps: int = 40,
        seed: int = 7,
    ):
        self.task_spec = task_spec
        self.tasks = tasks
        self.dependencies = dependencies
        self.max_steps = max_steps
        self.rng_seed = seed
        set_seed(seed)

        self.subtask_ids = [t.task_id for t in tasks]
        self.subtask_index = {sid: idx for idx, sid in enumerate(self.subtask_ids)}
        candidate_counts = {len(t.candidates) for t in tasks}
        if len(candidate_counts) != 1:
            raise ValueError("All tasks must have the same number of candidates")
        self.num_candidates = candidate_counts.pop()

        self.graph = SubtaskGraph(
            nodes=[
                SubtaskNode(
                    subtask_id=s.subtask_id,
                    description=s.description,
                    assigned_agent=s.assigned_agent,
                )
                for s in task_spec.subtasks
            ],
            edges=dependencies,
        )
        self.graph_state = GraphState(self.graph)
        self.step_index = 0
        self.last_actions: Dict[int, int] = {
            i: 0 for i in range(task_spec.agent_count)
        }

    def reset(self) -> Dict[int, Observation]:
        self.graph_state = GraphState(self.graph)
        self.step_index = 0
        self.last_actions = {i: 0 for i in range(self.task_spec.agent_count)}
        return self._build_obs()

    def get_task_spec(self) -> TaskSpec:
        return self.task_spec

    @property
    def action_size(self) -> int:
        return len(self.subtask_ids) * self.num_candidates + 1

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
            decoded = self._decode_action(action)
            if decoded is None:
                invalid_actions.append(agent_id)
                continue
            task_id, candidate_idx = decoded
            if self.graph_state.is_completed(task_id):
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            if not self._deps_satisfied(task_id):
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            assigned = self.graph.nodes[task_id].assigned_agent
            if assigned is not None and assigned != agent_id:
                invalid_actions.append(agent_id)
                if self.last_actions.get(agent_id) == action:
                    repeated_actions.append(agent_id)
                continue
            task = self.tasks[self.subtask_index[task_id]]
            if task.evaluate(candidate_idx):
                completed_this_step.append(task_id)
            if self.last_actions.get(agent_id) == action:
                repeated_actions.append(agent_id)

        for task_id in completed_this_step:
            self.graph_state.mark_completed(task_id)

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

    def _decode_action(self, action: int) -> Optional[Tuple[str, int]]:
        idx = action - 1
        if idx < 0:
            return None
        task_idx = idx // self.num_candidates
        candidate_idx = idx % self.num_candidates
        if task_idx < 0 or task_idx >= len(self.subtask_ids):
            return None
        return self.subtask_ids[task_idx], candidate_idx

    def _deps_satisfied(self, task_id: str) -> bool:
        preds = self.graph.predecessors(task_id)
        return preds.issubset(self.graph_state.completed)

    def _has_available_task(self, agent_id: int) -> bool:
        for task_id in self.subtask_ids:
            if self.graph_state.is_completed(task_id):
                continue
            if not self._deps_satisfied(task_id):
                continue
            assigned = self.graph.nodes[task_id].assigned_agent
            if assigned is None or assigned == agent_id:
                return True
        return False

    def _build_obs(self) -> Dict[int, Observation]:
        obs_by_agent: Dict[int, Observation] = {}
        for agent_id in range(self.task_spec.agent_count):
            completed_mask, _ = self.graph_state.masks()
            available_mask = []
            for task_id in self.subtask_ids:
                if self.graph_state.is_completed(task_id):
                    available_mask.append(0)
                    continue
                if not self._deps_satisfied(task_id):
                    available_mask.append(0)
                    continue
                assigned = self.graph.nodes[task_id].assigned_agent
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
