"""
Compare GRAFT multi-agent vs a CoMLRL-style collaborative baseline on code generation tasks.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from graft.bandit import Exp3Bandit
from graft.benchmarks.comlrl_utils import (
    check_aux_call_without_assignment,
    check_aux_function_usage,
    check_function_definition,
    check_syntax,
    combine_code,
    extract_specific_function,
    is_wrapper_function,
)
from graft.benchmarks.real_code_tasks import CodeTask, build_real_code_tasks, evaluate_code
from graft.failure import FailureLabels, FailureTaxonomy, ProcessRewardShaper
from graft.graphs import SubtaskGraph, SubtaskNode
from graft.planner import MistralLLMClient
from graft.utils import set_seed


@dataclass
class TaskMetric:
    task_id: str
    success: bool
    passed: int
    total: int
    status: str
    llm_calls: int
    comm_messages: int
    graph_id: Optional[str] = None
    comlrl_reward: Optional[float] = None


def load_mistral_config() -> tuple[str, str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Copy .env.example to .env and add your key."
        )
    model = (
        os.getenv("MISTRAL_MODEL")
        or os.getenv("MISRAL_MODEL")
        or "mistral-small-latest"
    )
    endpoint = (
        os.getenv("MISTRAL_ENDPOINT")
        or os.getenv("MISRAL_ENDPOINT")
        or "https://api.mistral.ai/v1/chat/completions"
    )
    return api_key, model, endpoint


def extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    start = text.find("```")
    rest = text[start + 3 :]
    if rest.startswith("python"):
        rest = rest[len("python") :]
    end = rest.find("```")
    if end == -1:
        return rest.strip()
    return rest[:end].strip()


def build_graph_ensemble() -> List[Tuple[str, SubtaskGraph]]:
    nodes_full = [
        SubtaskNode("analyze", "Analyze the problem and constraints.", assigned_agent=0),
        SubtaskNode("implement", "Write the solution code.", assigned_agent=1),
        SubtaskNode("review", "Review and fix the solution.", assigned_agent=2),
    ]
    graph_full = SubtaskGraph(
        nodes_full,
        edges=[("analyze", "implement"), ("implement", "review")],
    )
    nodes_fast = [
        SubtaskNode("implement", "Write the solution code.", assigned_agent=1),
        SubtaskNode("review", "Review and fix the solution.", assigned_agent=2),
    ]
    graph_fast = SubtaskGraph(nodes_fast, edges=[("implement", "review")])
    nodes_min = [
        SubtaskNode("implement", "Write the solution code.", assigned_agent=1),
    ]
    graph_min = SubtaskGraph(nodes_min, edges=[])
    return [("full", graph_full), ("fast", graph_fast), ("min", graph_min)]


def prompt_for_node(task: CodeTask, node_id: str, context: List[str]) -> str:
    header = "You are an expert Python developer."
    if node_id == "analyze":
        return (
            f"{header}\n\nTask:\n{task.prompt()}\n\n"
            "Provide a concise plan and edge cases. No code."
        )
    if node_id == "implement":
        context_text = "\n".join(context)
        return (
            f"{header}\n\nTask:\n{task.prompt()}\n\n"
            f"Context:\n{context_text}\n\n"
            "Return only valid Python code."
        )
    if node_id == "review":
        context_text = "\n".join(context)
        return (
            f"{header}\n\nTask:\n{task.prompt()}\n\n"
            f"Existing solution:\n{context_text}\n\n"
            "Fix any issues and return only corrected Python code."
        )
    return f"{header}\n\nTask:\n{task.prompt()}\n\nReturn only valid Python code."


def run_graft(
    client: MistralLLMClient,
    task: CodeTask,
    graph: SubtaskGraph,
) -> Tuple[str, int, int]:
    outputs: Dict[str, str] = {}
    llm_calls = 0
    comm_messages = 0
    order = graph.node_order()
    for node_id in order:
        preds = graph.predecessors(node_id)
        context = []
        for pred in preds:
            comm_messages += 1
            context.append(outputs.get(pred, ""))
        prompt = prompt_for_node(task, node_id, context)
        response = client.generate(prompt)
        outputs[node_id] = extract_code(response)
        llm_calls += 1
    final_node = order[-1]
    return outputs[final_node], llm_calls, comm_messages


def run_comlrl(
    client: MistralLLMClient,
    task: CodeTask,
) -> Tuple[str, str, int, int]:
    aux_prompt = (
        "Write a helper function named aux that will help solve the task. "
        "Return only valid Python code for aux."
    )
    main_prompt = (
        f"{task.prompt()}\n\n"
        "You may use a helper function named aux. "
        "Return only valid Python code for the main function."
    )
    aux_code = extract_code(client.generate(f"{task.prompt()}\n\n{aux_prompt}"))
    main_code = extract_code(client.generate(main_prompt))
    llm_calls = 2
    comm_messages = 1
    return aux_code, main_code, llm_calls, comm_messages


def comlrl_reward(task: CodeTask, aux_code: str, main_code: str) -> float:
    reward = 0.0
    aux_ok, _ = check_function_definition(aux_code, "aux")
    if aux_ok:
        reward += 0.4
    main_ok, _ = check_function_definition(main_code, task.function_name)
    if main_ok:
        reward += 0.6
    combined = combine_code(aux_code, main_code)
    syntax_ok, _ = check_syntax(combined)
    if syntax_ok:
        reward += 0.5
    result = evaluate_code(task, combined)
    if result.total > 0:
        reward += (result.passed / float(result.total)) * 1.0
    main_func = extract_specific_function(main_code, task.function_name)
    if result.passed > 0 and aux_ok and check_aux_function_usage(main_func):
        reward += 0.5
        if not is_wrapper_function(main_func):
            reward += 1.0
        if check_aux_call_without_assignment(main_func):
            reward -= 0.5
    return reward


def build_failure_labels(passed: int, total: int, status: str) -> FailureLabels:
    progress = 0.0 if total == 0 else passed / float(total)
    flags = {
        "compile_error": 1.0 if status == "compile_error" else 0.0,
        "runtime_error": 1.0 if status == "runtime_error" else 0.0,
        "missing_function": 1.0 if status == "missing_function" else 0.0,
        "timeout": 1.0 if status == "timeout" else 0.0,
    }
    return FailureLabels(progress=progress, flags=flags)


def aggregate_metrics(metrics: List[TaskMetric]) -> Dict[str, float]:
    total = len(metrics)
    success = sum(1 for m in metrics if m.success)
    avg_passed = sum(m.passed for m in metrics) / max(total, 1)
    avg_total = sum(m.total for m in metrics) / max(total, 1)
    avg_pass_rate = avg_passed / max(avg_total, 1)
    avg_calls = sum(m.llm_calls for m in metrics) / max(total, 1)
    avg_comm = sum(m.comm_messages for m in metrics) / max(total, 1)
    avg_comlrl = (
        sum(m.comlrl_reward for m in metrics if m.comlrl_reward is not None)
        / max(total, 1)
    )
    return {
        "tasks": total,
        "success_rate": success / max(total, 1),
        "avg_test_pass_rate": avg_pass_rate,
        "avg_llm_calls": avg_calls,
        "avg_comm_messages": avg_comm,
        "avg_comlrl_reward": avg_comlrl,
    }


def write_metrics(path: Path, metrics: List[TaskMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "success",
                "passed",
                "total",
                "status",
                "llm_calls",
                "comm_messages",
                "graph_id",
                "comlrl_reward",
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m.task_id,
                    int(m.success),
                    m.passed,
                    m.total,
                    m.status,
                    m.llm_calls,
                    m.comm_messages,
                    m.graph_id or "",
                    f"{m.comlrl_reward:.3f}" if m.comlrl_reward is not None else "",
                ]
            )


def plot_metrics(
    output_path: Path,
    graft_summary: Dict[str, float],
    comlrl_summary: Dict[str, float],
) -> None:
    labels = ["success_rate", "avg_test_pass_rate", "avg_llm_calls"]
    graft_vals = [graft_summary[l] for l in labels]
    comlrl_vals = [comlrl_summary[l] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].bar(["GRAFT", "CoMLRL"], [graft_vals[0], comlrl_vals[0]], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(["GRAFT", "CoMLRL"], [graft_vals[1], comlrl_vals[1]], color=["#4C78A8", "#F58518"])
    axes[1].set_title("Avg Test Pass Rate")
    axes[1].set_ylim(0, 1)

    axes[2].bar(["GRAFT", "CoMLRL"], [graft_vals[2], comlrl_vals[2]], color=["#4C78A8", "#F58518"])
    axes[2].set_title("Avg LLM Calls")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    set_seed(7)
    tasks = build_real_code_tasks()
    api_key, model, endpoint = load_mistral_config()
    client = MistralLLMClient(api_key=api_key, model=model, endpoint=endpoint)

    graphs = build_graph_ensemble()
    bandit = Exp3Bandit(num_arms=len(graphs), gamma=0.2, seed=7)

    taxonomy = FailureTaxonomy(["compile_error", "runtime_error", "missing_function", "timeout"])
    shaper = ProcessRewardShaper(
        progress_weight=1.0,
        failure_weights={
            "compile_error": 0.7,
            "runtime_error": 0.5,
            "missing_function": 0.7,
            "timeout": 1.0,
        },
        taxonomy=taxonomy,
    )

    graft_metrics: List[TaskMetric] = []
    comlrl_metrics: List[TaskMetric] = []

    for task in tasks:
        graph_idx = bandit.sample()
        graph_id, graph = graphs[graph_idx]
        graft_code, graft_calls, graft_comm = run_graft(client, task, graph)
        graft_result = evaluate_code(task, graft_code)
        graft_labels = build_failure_labels(
            graft_result.passed, graft_result.total, graft_result.status
        )
        graft_reward = shaper.compute(graft_labels)
        bandit.update(graph_idx, max(0.0, min(1.0, graft_reward)))
        graft_metrics.append(
            TaskMetric(
                task_id=task.task_id,
                success=graft_result.success,
                passed=graft_result.passed,
                total=graft_result.total,
                status=graft_result.status,
                llm_calls=graft_calls,
                comm_messages=graft_comm,
                graph_id=graph_id,
            )
        )

        aux_code, main_code, com_calls, com_comm = run_comlrl(client, task)
        combined = combine_code(aux_code, main_code)
        com_result = evaluate_code(task, combined)
        reward = comlrl_reward(task, aux_code, main_code)
        comlrl_metrics.append(
            TaskMetric(
                task_id=task.task_id,
                success=com_result.success,
                passed=com_result.passed,
                total=com_result.total,
                status=com_result.status,
                llm_calls=com_calls,
                comm_messages=com_comm,
                comlrl_reward=reward,
            )
        )

    graft_summary = aggregate_metrics(graft_metrics)
    comlrl_summary = aggregate_metrics(comlrl_metrics)

    runs_dir = Path(__file__).resolve().parents[1] / "runs" / "graft_vs_comlrl"
    write_metrics(runs_dir / "graft_metrics.csv", graft_metrics)
    write_metrics(runs_dir / "comlrl_metrics.csv", comlrl_metrics)
    plot_metrics(runs_dir / "comparison.png", graft_summary, comlrl_summary)

    with (runs_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"graft": graft_summary, "comlrl": comlrl_summary},
            handle,
            indent=2,
        )

    print("GRAFT summary:", graft_summary)
    print("CoMLRL summary:", comlrl_summary)


if __name__ == "__main__":
    main()
