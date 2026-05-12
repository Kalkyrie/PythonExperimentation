"""
Tutorial 04 - Scorers
======================
Scorers judge the model's output and return a Score.

This tutorial covers:
  1. exact()            - string equality (case-insensitive)
  2. includes()         - target appears anywhere in the output
  3. pattern()          - regex match
  4. model_graded_fact()- use another LLM to judge factual correctness
  5. Writing a custom scorer

Run:
    inspect eval 04_scorers.py --model mockllm/model
    inspect eval 04_scorers.py@graded --model anthropic/claude-haiku-4-5-20251001
"""

import re

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    exact,
    includes,
    model_graded_fact,
    pattern,
    scorer,
)
from inspect_ai.solver import TaskState, generate


FACTUAL_SAMPLES = [
    Sample(input="What is the capital of France?",   target="Paris"),
    Sample(input="What is the capital of Germany?",  target="Berlin"),
    Sample(input="What year did WW2 end?",            target="1945"),
]

NUMBER_SAMPLES = [
    Sample(input="What is 12 * 12?", target="144"),
    Sample(input="What is 7 + 8?",   target="15"),
]


# ---------------------------------------------------------------------------
# 1. exact() — model output must equal target (case-insensitive, stripped)
# ---------------------------------------------------------------------------
@task
def with_exact():
    return Task(dataset=FACTUAL_SAMPLES, solver=generate(), scorer=exact())


# ---------------------------------------------------------------------------
# 2. includes() — target must appear somewhere in the model output
# ---------------------------------------------------------------------------
@task
def with_includes():
    return Task(dataset=FACTUAL_SAMPLES, solver=generate(), scorer=includes())


# ---------------------------------------------------------------------------
# 3. pattern() — target is a regex; the model output must match
# ---------------------------------------------------------------------------
@task
def with_pattern():
    # Match a number anywhere in the response
    return Task(dataset=NUMBER_SAMPLES, solver=generate(), scorer=pattern(r"\d+"))


# ---------------------------------------------------------------------------
# 4. model_graded_fact() — a grader LLM judges whether the answer is correct
#    Requires a real model; with mockllm it always returns a fixed response.
# ---------------------------------------------------------------------------
@task
def graded():
    return Task(dataset=FACTUAL_SAMPLES, solver=generate(), scorer=model_graded_fact())


# ---------------------------------------------------------------------------
# 5. Custom scorer using @scorer decorator
#    Returns CORRECT if the model output contains a number, otherwise INCORRECT
# ---------------------------------------------------------------------------
@scorer(metrics=[accuracy()])
def contains_number():
    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion
        has_number = bool(re.search(r"\d", output))
        return Score(
            value="C" if has_number else "I",
            answer=output,
            explanation="Output contains a digit" if has_number else "No digit found",
        )

    return score


@task
def with_custom_scorer():
    return Task(dataset=NUMBER_SAMPLES, solver=generate(), scorer=contains_number())


if __name__ == "__main__":
    tasks = [with_exact, with_includes, with_pattern, with_custom_scorer]
    for t in tasks:
        results = eval(t(), model="mockllm/model", display="none")
        r = results[0]
        print(f"{r.task}: {r.results.scores[0].metrics}")
