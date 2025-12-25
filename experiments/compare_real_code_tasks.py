"""
Compare single-agent vs multi-agent performance on real coding tasks using GRAFT.
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
from graft.benchmarks.real_code_tasks import CodeTask, EvalResult, build_real_code_tasks, evaluate_code
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
    if start == -1:
        return text.strip()
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
    graph_fast = SubtaskGraph(
        nodes_fast,
        edges=[("implement", "review")],
    )
    nodes_min = [
        SubtaskNode("implement", "Write the solution code.", assigned_agent=1),
    ]
    graph_min = SubtaskGraph(nodes_min, edges=[])
    return [
        ("full", graph_full),
        ("fast", graph_fast),
        ("min", graph_min),
    ]


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


def run_single_agent(
    client: MistralLLMClient, task: CodeTask
) -> Tuple[str, int]:
    prompt = prompt_for_node(task, "implement", [])
    response = client.generate(prompt)
    return extract_code(response), 1


def run_multi_agent(
    client: MistralLLMClient,
    task: CodeTask,
    graph: SubtaskGraph,
) -> Tuple[str, int, int]:
    outputs: Dict[str, str] = {}
    llm_calls = 0
    comm_messages = 0
    # Topological order based on insertion order and dependencies.
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


def build_failure_labels(result: EvalResult) -> FailureLabels:
    progress = 0.0 if result.total == 0 else result.passed / float(result.total)
    flags = {
        "compile_error": 1.0 if result.status == "compile_error" else 0.0,
        "runtime_error": 1.0 if result.status == "runtime_error" else 0.0,
        "missing_function": 1.0 if result.status == "missing_function" else 0.0,
        "timeout": 1.0 if result.status == "timeout" else 0.0,
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
    return {
        "tasks": total,
        "success_rate": success / max(total, 1),
        "avg_tests_passed": avg_passed,
        "avg_test_pass_rate": avg_pass_rate,
        "avg_llm_calls": avg_calls,
        "avg_comm_messages": avg_comm,
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
                ]
            )


def plot_metrics(
    output_path: Path,
    single_summary: Dict[str, float],
    multi_summary: Dict[str, float],
) -> None:
    labels = ["success_rate", "avg_test_pass_rate"]
    single_vals = [single_summary[l] for l in labels]
    multi_vals = [multi_summary[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(["single", "multi"], [single_vals[0], multi_vals[0]], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Task Success Rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(["single", "multi"], [single_vals[1], multi_vals[1]], color=["#4C78A8", "#F58518"])
    axes[1].set_title("Avg Test Pass Rate")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    set_seed(7)
    tasks = build_real_code_tasks()
    api_key, model, endpoint = load_mistral_config()
    client = MistralLLMClient(api_key=api_key, model=model, endpoint=endpoint)

    # Failure-aware reward shaping for graph selection.
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

    graphs = build_graph_ensemble()
    bandit = Exp3Bandit(num_arms=len(graphs), gamma=0.2, seed=7)

    single_metrics: List[TaskMetric] = []
    multi_metrics: List[TaskMetric] = []

    for task in tasks:
        # Single-agent run.
        code, calls = run_single_agent(client, task)
        result = evaluate_code(task, code)
        single_metrics.append(
            TaskMetric(
                task_id=task.task_id,
                success=result.success,
                passed=result.passed,
                total=result.total,
                status=result.status,
                llm_calls=calls,
                comm_messages=0,
            )
        )

        # Multi-agent run using GRAFT graph selection.
        graph_idx = bandit.sample()
        graph_id, graph = graphs[graph_idx]
        code, calls, comm = run_multi_agent(client, task, graph)
        result = evaluate_code(task, code)
        labels = build_failure_labels(result)
        shaped_reward = shaper.compute(labels)
        reward = max(0.0, min(1.0, shaped_reward))
        bandit.update(graph_idx, reward)

        multi_metrics.append(
            TaskMetric(
                task_id=task.task_id,
                success=result.success,
                passed=result.passed,
                total=result.total,
                status=result.status,
                llm_calls=calls,
                comm_messages=comm,
                graph_id=graph_id,
            )
        )

    single_summary = aggregate_metrics(single_metrics)
    multi_summary = aggregate_metrics(multi_metrics)

    runs_dir = Path(__file__).resolve().parents[1] / "runs" / "real_code"
    write_metrics(runs_dir / "single_agent_tasks.csv", single_metrics)
    write_metrics(runs_dir / "multi_agent_tasks.csv", multi_metrics)
    plot_metrics(runs_dir / "comparison.png", single_summary, multi_summary)

    with (runs_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"single": single_summary, "multi": multi_summary},
            handle,
            indent=2,
        )

    print("Single summary:", single_summary)
    print("Multi summary:", multi_summary)


if __name__ == "__main__":
    main()
