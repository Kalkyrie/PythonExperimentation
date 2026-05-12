# Inspect Tutorial — Custom Learning Materials

A hands-on tutorial series for [Inspect](https://inspect.aisi.org.uk/), the open-source LLM evaluation framework by the UK AI Security Institute (AISI).

---

## What is Inspect?

Inspect lets you define, run, and analyse evaluations ("evals") of large language models. An eval has three parts:

```
Dataset  →  Solver(s)  →  Scorer
(inputs)    (model calls)  (judgement)
```

It is the primary tool used by AISI to safety-test frontier AI models.

---

## Quick Install

```bash
pip install inspect-ai
# For Anthropic models:
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
```

---

## Tutorial Files

| File | Concepts |
|------|----------|
| `01_hello_inspect.py` | Minimal task, `@task`, `generate()`, `exact()` |
| `02_datasets.py` | Inline, CSV, and JSON datasets |
| `03_solvers.py` | `system_message`, `prompt_template`, `chain_of_thought`, custom solver |
| `04_scorers.py` | `exact`, `includes`, `pattern`, `model_graded_fact`, custom scorer |
| `05_tools_and_agents.py` | `@tool`, `basic_agent()`, agentic loops |
| `06_safety_eval_example.py` | Refusal detection, policy compliance, keyword scoring |

---

## Running the Tutorials

### With a real model (recommended)

```bash
# Anthropic
inspect eval 01_hello_inspect.py --model anthropic/claude-haiku-4-5-20251001

# OpenAI
inspect eval 01_hello_inspect.py --model openai/gpt-4o-mini

# Limit to first 5 samples (useful for quick tests)
inspect eval 02_datasets.py --model anthropic/claude-haiku-4-5-20251001 --limit 5
```

### Without an API key (offline, mock model)

All tutorials can be run with the built-in mock model for structure validation:

```bash
inspect eval 01_hello_inspect.py --model mockllm/model
```

The mock model returns a fixed string, so scores won't be meaningful — but it confirms the eval pipeline runs correctly.

### Run all tutorials at once

```bash
inspect eval-set 01_hello_inspect.py 02_datasets.py 03_solvers.py 04_scorers.py \
  --model anthropic/claude-haiku-4-5-20251001
```

---

## Viewing Results

Inspect saves logs to `./logs/` by default. Open the visual viewer with:

```bash
inspect view
```

Or point it at a specific log directory:

```bash
inspect view --log-dir ./logs
```

---

## Key CLI Options

| Flag | Purpose |
|------|---------|
| `--model` | Provider/model string, e.g. `anthropic/claude-haiku-4-5-20251001` |
| `--limit N` | Run only the first N samples |
| `--log-dir PATH` | Where to save results (default: `./logs`) |
| `--temperature F` | Override model temperature |
| `--max-connections N` | Parallel API connections |

---

## Core Concepts Cheat Sheet

```python
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, csv_dataset, json_dataset
from inspect_ai.solver import generate, system_message, chain_of_thought, basic_agent, solver
from inspect_ai.scorer import exact, includes, pattern, model_graded_fact, scorer, Score
from inspect_ai.tool import tool

# Minimal task
@task
def my_eval():
    return Task(
        dataset=[Sample(input="...", target="...")],
        solver=generate(),
        scorer=exact(),
    )

# Run programmatically
results = eval(my_eval(), model="anthropic/claude-haiku-4-5-20251001")
```

---

## Supported Model Providers

| Provider | Model string prefix | Package |
|----------|---------------------|---------|
| Anthropic | `anthropic/` | `pip install anthropic` |
| OpenAI | `openai/` | `pip install openai` |
| AWS Bedrock | `bedrock/` | built-in |
| Ollama (local) | `ollama/` | install Ollama |
| OpenAI-compatible | `openai/` + custom base URL | — |
| Mock (testing) | `mockllm/model` | built-in |

Full list: https://inspect.aisi.org.uk/providers.html

---

## Further Reading

- [Official Docs](https://inspect.aisi.org.uk/)
- [inspect_evals — 200+ pre-built evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
