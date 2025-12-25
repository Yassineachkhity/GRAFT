"""
Realistic coding tasks with evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CodeTask:
    task_id: str
    description: str
    function_name: str
    tests: List[Tuple[Tuple[Any, ...], Any]]
    signature: str
    constraints: Optional[str] = None

    def prompt(self) -> str:
        prompt = f"Implement a Python function:\n{self.signature}\n\nTask:\n{self.description}\n"
        if self.constraints:
            prompt += f"\nConstraints:\n{self.constraints}\n"
        prompt += "\nReturn only valid Python code."
        return prompt


@dataclass
class EvalResult:
    status: str
    passed: int
    total: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.passed == self.total and self.total > 0


def build_real_code_tasks() -> List[CodeTask]:
    return [
        CodeTask(
            task_id="palindrome",
            description="Return True if the input string is a palindrome (case-sensitive).",
            function_name="is_palindrome",
            signature="def is_palindrome(s: str) -> bool:",
            tests=[
                (("racecar",), True),
                (("abba",), True),
                (("abc",), False),
                (("",), True),
            ],
        ),
        CodeTask(
            task_id="two_sum",
            description=(
                "Given a list of integers and a target, return a tuple of two indices "
                "i, j (i < j) such that nums[i] + nums[j] == target. "
                "Assume exactly one solution exists."
            ),
            function_name="two_sum",
            signature="def two_sum(nums: list[int], target: int) -> tuple[int, int]:",
            tests=[
                (([2, 7, 11, 15], 9), (0, 1)),
                (([3, 2, 4], 6), (1, 2)),
                (([3, 3], 6), (0, 1)),
            ],
        ),
        CodeTask(
            task_id="valid_parentheses",
            description=(
                "Return True if the string has valid parentheses. "
                "Characters are only '()[]{}'."
            ),
            function_name="is_valid",
            signature="def is_valid(s: str) -> bool:",
            tests=[
                (("()",), True),
                (("()[]{}",), True),
                (("(]",), False),
                (("([{}])",), True),
            ],
        ),
        CodeTask(
            task_id="fizzbuzz",
            description=(
                "Return a list of strings from 1 to n. "
                "Use 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
                "and 'FizzBuzz' for multiples of both."
            ),
            function_name="fizzbuzz",
            signature="def fizzbuzz(n: int) -> list[str]:",
            tests=[
                ((5,), ["1", "2", "Fizz", "4", "Buzz"]),
                ((1,), ["1"]),
                ((15,), [
                    "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
                    "11", "Fizz", "13", "14", "FizzBuzz"
                ]),
            ],
        ),
        CodeTask(
            task_id="rotate_array",
            description=(
                "Rotate the list to the right by k steps and return the rotated list. "
                "Do not modify the input list."
            ),
            function_name="rotate_right",
            signature="def rotate_right(nums: list[int], k: int) -> list[int]:",
            tests=[
                (([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3]),
                (([1, 2], 3), [2, 1]),
                (([], 4), []),
            ],
        ),
        CodeTask(
            task_id="merge_intervals",
            description=(
                "Given a list of intervals [start, end], merge overlapping intervals "
                "and return the merged list sorted by start."
            ),
            function_name="merge_intervals",
            signature="def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:",
            tests=[
                (([(1, 3), (2, 6), (8, 10), (15, 18)],), [(1, 6), (8, 10), (15, 18)]),
                (([(1, 4), (4, 5)],), [(1, 5)]),
                (([],), []),
            ],
        ),
    ]


def evaluate_code(task: CodeTask, code: str, timeout_s: int = 3) -> EvalResult:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_eval, args=(task, code, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return EvalResult(status="timeout", passed=0, total=len(task.tests), error="timeout")
    try:
        result = queue.get_nowait()
    except Empty:
        return EvalResult(status="error", passed=0, total=len(task.tests), error="no result")
    return result


def _worker_eval(task: CodeTask, code: str, queue) -> None:
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as exc:  # noqa: BLE001
        queue.put(
            EvalResult(
                status="compile_error",
                passed=0,
                total=len(task.tests),
                error=str(exc),
            )
        )
        return
    func = namespace.get(task.function_name)
    if not callable(func):
        queue.put(
            EvalResult(
                status="missing_function",
                passed=0,
                total=len(task.tests),
                error=f"{task.function_name} not found",
            )
        )
        return
    passed = 0
    try:
        for args, expected in task.tests:
            output = func(*args)
            if output == expected:
                passed += 1
        status = "ok" if passed == len(task.tests) else "failed_tests"
        queue.put(
            EvalResult(
                status=status,
                passed=passed,
                total=len(task.tests),
            )
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(
            EvalResult(
                status="runtime_error",
                passed=passed,
                total=len(task.tests),
                error=str(exc),
            )
        )
