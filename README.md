# GRAFT: Failure-Aware Graph Ensembles for LLM-Guided MARL

This is a clean, well-documented reference implementation of the GRAFT framework
described in "Failure-Aware Graph Ensembles for Reliable LLM-Guided Multi-Agent
Reinforcement Learning." It focuses on clarity and modularity so you can swap in
your own planners, judges, MARL algorithms, and environments.

Core ideas implemented:
- Plan ensemble + online graph selection (Exp3 bandit)
- Failure-aware process rewards + critic distillation
- Sparse, graph-aligned communication with gating

This repo ships with a toy multi-agent environment and a runnable example. The
LLM planner is optional: a mock planner is included, and a local HF-based adapter
is provided for plugging in an open-source model installed on your machine.

## Project layout

- `graft/` core framework modules
  - `graphs.py` subtask graphs and graph state
  - `planner.py` planner interfaces + mock + local HF adapter
  - `bandit.py` Exp3 implementation
  - `failure.py` failure taxonomy, judge, process reward, distillation
  - `communication.py` graph-aligned messaging + gating
  - `marl.py` MARL algorithm interface + simple independent Q-learner
  - `training.py` GRAFT training loop
  - `envs/toy_env.py` toy multi-agent environment
  - `config.py` dataclass configs
  - `utils.py` helpers
- `examples/run_toy.py` minimal runnable demo

## Quickstart (toy demo)

1) Install dependencies

```
pip install -r requirements.txt
```

2) Configure Gemini

Copy the env template and add your API key:

```
copy .env.example .env
```

Edit `.env` to set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (default:
`gemini-2.5-flash`).

3) Run the toy demo

```
python examples/run_toy.py
```

## LLM planner notes

`graft/planner.py` includes:
- `MockPlanner` for offline development and testing
- `LocalLLMPlanner` which expects a local HF model path and uses `transformers`
  if installed. This is optional and not required for the toy demo.
- `GeminiLLMClient` for using the Gemini API via `google-generativeai`

## Extending

- Replace `MockPlanner` with your own planner.
- Replace `HeuristicFailureJudge` with an LLM or tool-based judge.
- Swap in your MARL algorithm inside `marl.py` while keeping the GRAFT trainer.

## Experiments

For a simple single-agent vs multi-agent comparison on coding tasks, run:

```
python experiments/compare_code_tasks.py
```

This script logs metrics to `runs/` and saves a comparison plot.
