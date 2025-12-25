"""
Planner interfaces and implementations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from graft.graphs import SubtaskGraph, SubtaskNode
from graft.types import TaskSpec
from graft.utils import set_seed


class Planner:
    """Base planner interface."""

    def plan(self, task: TaskSpec) -> List[SubtaskGraph]:
        raise NotImplementedError


class MockPlanner(Planner):
    """Generates an ensemble by perturbing the ground-truth dependency graph."""

    def __init__(self, ensemble_size: int = 4, noise_prob: float = 0.25, seed: int = 7):
        self.ensemble_size = ensemble_size
        self.noise_prob = noise_prob
        self.seed = seed
        set_seed(seed)

    def plan(self, task: TaskSpec) -> List[SubtaskGraph]:
        base = self._graph_from_task(task)
        graphs = [base]
        for _ in range(self.ensemble_size - 1):
            graphs.append(self._mutate_graph(base))
        return graphs

    def _graph_from_task(self, task: TaskSpec) -> SubtaskGraph:
        nodes = [
            SubtaskNode(
                subtask_id=s.subtask_id,
                description=s.description,
                assigned_agent=s.assigned_agent,
            )
            for s in task.subtasks
        ]
        return SubtaskGraph(nodes, task.dependencies)

    def _mutate_graph(self, graph: SubtaskGraph) -> SubtaskGraph:
        nodes = list(graph.nodes.values())
        edges = list(graph.edges)
        # Randomly drop edges
        kept = []
        for edge in edges:
            if self._rand() > self.noise_prob:
                kept.append(edge)
        edges = kept
        # Randomly add edges that do not introduce cycles
        node_ids = list(graph.nodes.keys())
        for _ in range(len(node_ids)):
            if self._rand() < self.noise_prob:
                u = self._choice(node_ids)
                v = self._choice(node_ids)
                if u == v:
                    continue
                candidate = (u, v)
                if candidate in edges:
                    continue
                try:
                    _ = SubtaskGraph(nodes, edges + [candidate])
                    edges.append(candidate)
                except ValueError:
                    continue
        return SubtaskGraph(nodes, edges)

    def _rand(self) -> float:
        import random

        return random.random()

    def _choice(self, items: Sequence[str]) -> str:
        import random

        return random.choice(list(items))


class LLMClient:
    """Simple LLM client interface."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class HFLocalLLMClient(LLMClient):
    """
    Local HuggingFace client for an open-source model.
    This expects `transformers` installed and a local model path.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 512):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for HFLocalLLMClient"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class GeminiLLMClient(LLMClient):
    """
    Gemini client using google-generativeai with an API key from the environment.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        fallback_models: Optional[Sequence[str]] = None,
        response_mime_type: str = "application/json",
    ):
        try:
            import google.generativeai as genai
            from google.api_core import exceptions as gexc
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for GeminiLLMClient"
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai
        self._exceptions = gexc
        self.model_name = model
        self.fallback_models = list(
            fallback_models
            or [
                "gemini-2.5-flash-latest",
                "gemini-2.0-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.0-pro",
            ]
        )
        self.response_mime_type = response_mime_type

    def generate(self, prompt: str) -> str:
        model_names = [self.model_name] + [
            name for name in self.fallback_models if name != self.model_name
        ]
        last_exc: Optional[Exception] = None
        for model_name in model_names:
            try:
                return self._generate_with_model(model_name, prompt, use_mime=True)
            except Exception as exc:
                if self._is_mime_error(exc):
                    try:
                        return self._generate_with_model(
                            model_name, prompt, use_mime=False
                        )
                    except Exception as exc2:
                        if self._is_not_found(exc2):
                            last_exc = exc2
                            continue
                        raise
                if self._is_not_found(exc):
                    last_exc = exc
                    continue
                raise
        if last_exc:
            raise last_exc
        raise ValueError("Gemini response contained no text")

    def _generate_with_model(self, model_name: str, prompt: str, use_mime: bool) -> str:
        model = self._genai.GenerativeModel(model_name)
        if use_mime:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": self.response_mime_type},
            )
        else:
            response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        raise ValueError("Gemini response contained no text")

    def _is_not_found(self, exc: Exception) -> bool:
        return isinstance(exc, self._exceptions.NotFound) or "not found" in str(exc).lower()

    def _is_mime_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "response_mime_type" in text or "mime" in text


@dataclass
class LocalLLMPlanner(Planner):
    """
    Planner that asks a local LLM to return a JSON plan ensemble.
    """

    client: LLMClient
    ensemble_size: int = 4

    def plan(self, task: TaskSpec) -> List[SubtaskGraph]:
        prompt = self._build_prompt(task)
        raw = self.client.generate(prompt)
        payload = self._extract_json(raw)
        return self._parse_graphs(payload, task)

    def _build_prompt(self, task: TaskSpec) -> str:
        subtasks = "\n".join(
            f"- {s.subtask_id}: {s.description} (agent={s.assigned_agent})"
            for s in task.subtasks
        )
        deps = "\n".join(f"- {u} -> {v}" for u, v in task.dependencies) or "- none"
        return (
            "You are a planner. Produce a JSON array of candidate subtask graphs.\n"
            f"Return exactly {self.ensemble_size} graphs.\n\n"
            f"Task: {task.description}\n"
            f"Subtasks:\n{subtasks}\n"
            f"Known dependencies (optional):\n{deps}\n\n"
            "Use only the provided subtask ids. Do not invent new nodes.\n"
            "JSON format:\n"
            "[{\n"
            '  "nodes": [{"id": "t1", "desc": "...", "agent": 0}],\n'
            '  "edges": [{"from": "t1", "to": "t2"}]\n'
            "}]\n"
        )

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\\[.*\\]", text, re.DOTALL)
        if not match:
            raise ValueError("LLM output does not contain a JSON array")
        return match.group(0)

    def _parse_graphs(self, json_text: str, task: TaskSpec) -> List[SubtaskGraph]:
        data = json.loads(json_text)
        base_nodes = {
            s.subtask_id: SubtaskNode(
                subtask_id=s.subtask_id,
                description=s.description,
                assigned_agent=s.assigned_agent,
            )
            for s in task.subtasks
        }
        graphs = []
        for graph_dict in data:
            nodes = list(base_nodes.values())
            edges = []
            for edge in graph_dict.get("edges", []):
                src = edge.get("from")
                dst = edge.get("to")
                if src in base_nodes and dst in base_nodes:
                    edges.append((src, dst))
            try:
                graphs.append(SubtaskGraph(nodes, edges))
            except ValueError:
                graphs.append(SubtaskGraph(nodes, task.dependencies))
        if not graphs:
            raise ValueError("No graphs parsed from LLM output")
        return graphs
