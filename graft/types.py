"""
Shared data types used across the GRAFT framework.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SubtaskSpec:
    """Specifies a subtask for a task decomposition."""

    subtask_id: str
    description: str
    assigned_agent: Optional[int] = None


@dataclass
class TaskSpec:
    """Specifies a high-level task for the planner."""

    task_id: str
    description: str
    subtasks: List[SubtaskSpec]
    dependencies: List[Tuple[str, str]]
    agent_count: int


@dataclass
class Observation:
    """Minimal per-agent observation for the toy env and demo policies."""

    agent_id: int
    completed_mask: List[int]
    available_mask: List[int]
    last_action: int
    step_index: int
