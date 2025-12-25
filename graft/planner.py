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
    Gemini client using the Google GenAI SDK with an API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        fallback_models: Optional[Sequence[str]] = None,
        response_mime_type: str = "application/json",
        api_version: str = "v1",
    ):
        try:
            from google import genai
            from google.genai import types
            from google.api_core import exceptions as gexc
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiLLMClient"
            ) from exc
        self._genai = genai
        self._types = types
        self._exceptions = gexc
        http_options = self._types.HttpOptions(api_version=api_version)
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model_name = model
        self.fallback_models = list(
            fallback_models
            or [
                "gemini-2.5-flash-latest",
                "gemini-2.0-flash",
                "gemini-1.5-flash-latest",
            ]
        )
        self._model_list: Optional[List[str]] = None
        self.response_mime_type = response_mime_type

    def generate(self, prompt: str) -> str:
        model_names = [self.model_name] + [
            name for name in self.fallback_models if name != self.model_name
        ]
        for name in self._get_model_list():
            if name not in model_names:
                model_names.append(name)
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
                        if self._is_quota_exceeded(exc2):
                            raise exc2
                        if self._is_not_found(exc2):
                            last_exc = exc2
                            continue
                        raise
                if self._is_quota_exceeded(exc):
                    raise exc
                if self._is_not_found(exc):
                    last_exc = exc
                    continue
                raise
        if last_exc:
            raise last_exc
        raise ValueError("Gemini response contained no text")

    def _generate_with_model(self, model_name: str, prompt: str, use_mime: bool) -> str:
        if use_mime:
            config = self._types.GenerateContentConfig(
                response_mime_type=self.response_mime_type
            )
            response = self.client.models.generate_content(
                model=model_name, contents=prompt, config=config
            )
        else:
            response = self.client.models.generate_content(
                model=model_name, contents=prompt
            )
        text = getattr(response, "text", None)
        if text:
            return text
        raise ValueError("Gemini response contained no text")

    def _is_not_found(self, exc: Exception) -> bool:
        if isinstance(exc, self._exceptions.NotFound):
            return True
        return "not found" in str(exc).lower()

    def _is_quota_exceeded(self, exc: Exception) -> bool:
        if isinstance(exc, self._exceptions.ResourceExhausted):
            return True
        return "resource_exhausted" in str(exc).lower() or "quota" in str(exc).lower()

    def _get_model_list(self) -> List[str]:
        if self._model_list is not None:
            return self._model_list
        try:
            models = list(self.client.models.list())
        except Exception:
            self._model_list = []
            return self._model_list
        available = []
        for model in models:
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            name = getattr(model, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                available.append(name)
        self._model_list = available
        return self._model_list

    def _is_mime_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "response_mime_type" in text or "mime" in text


class MistralLLMClient(LLMClient):
    """
    Mistral client using the REST API with an API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        endpoint: str = "https://api.mistral.ai/v1/chat/completions",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required for MistralLLMClient") from exc
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        response = requests.post(
            self.endpoint, headers=headers, json=payload, timeout=self.timeout
        )
        if response.status_code != 200:
            raise ValueError(
                f"Mistral API error {response.status_code}: {response.text}"
            )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Mistral response contained no choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError("Mistral response contained no content")
        return content


@dataclass
class LocalLLMPlanner(Planner):
    """
    Planner that asks a local LLM to return a JSON plan ensemble.
    """

    client: LLMClient
    ensemble_size: int = 4
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        self._cache = {}

    def plan(self, task: TaskSpec) -> List[SubtaskGraph]:
        cache_key = None
        if self.cache_enabled:
            cache_key = self._cache_key(task)
            if cache_key in self._cache:
                return self._cache[cache_key]

        prompt = self._build_prompt(task)
        raw = self.client.generate(prompt)
        payload = self._extract_json(raw)
        graphs_payload = self._normalize_payload(payload)
        graphs = self._parse_graphs(graphs_payload, task)
        if self.cache_enabled and cache_key is not None:
            self._cache[cache_key] = graphs
        return graphs

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

    def _cache_key(self, task: TaskSpec) -> str:
        payload = {
            "task_id": task.task_id,
            "description": task.description,
            "subtasks": [
                {
                    "id": s.subtask_id,
                    "desc": s.description,
                    "agent": s.assigned_agent,
                }
                for s in task.subtasks
            ],
            "dependencies": sorted([list(dep) for dep in task.dependencies]),
            "ensemble_size": self.ensemble_size,
        }
        return json.dumps(payload, sort_keys=True)

    def _extract_json(self, text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char in "[{":
                try:
                    payload, _ = decoder.raw_decode(text[idx:])
                    return payload
                except json.JSONDecodeError:
                    continue
        raise ValueError("LLM output does not contain JSON data")

    def _normalize_payload(self, payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "graphs" in payload and isinstance(payload["graphs"], list):
                return payload["graphs"]
            if "plans" in payload and isinstance(payload["plans"], list):
                return payload["plans"]
            if "nodes" in payload:
                return [payload]
        raise ValueError("LLM JSON payload format not recognized")

    def _parse_graphs(self, data, task: TaskSpec) -> List[SubtaskGraph]:
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
