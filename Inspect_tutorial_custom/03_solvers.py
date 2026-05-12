"""
Tutorial 03 - Solvers
======================
Solvers are the "middle" of the evaluation pipeline.
They receive a TaskState and return a modified TaskState.

This tutorial covers:
  1. generate()          - single model call
  2. system_message()    - prepend a system prompt
  3. prompt_template()   - fill a Jinja-style template
  4. chain_of_thought()  - ask the model to reason step-by-step
  5. Writing a custom solver from scratch

Run:
    inspect eval 03_solvers.py --model mockllm/model
    inspect eval 03_solvers.py@with_system_message --model anthropic/claude-haiku-4-5-20251001
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain_of_thought,
    generate,
    prompt_template,
    solver,
    system_message,
)


SAMPLES = [
    Sample(input="Explain why the sky is blue in one sentence.", target="Rayleigh scattering"),
    Sample(input="What is the boiling point of water in Celsius?", target="100"),
]


# ---------------------------------------------------------------------------
# 1. Plain generate — just call the model with whatever is in the prompt
# ---------------------------------------------------------------------------
@task
def plain_generate():
    return Task(dataset=SAMPLES, solver=generate(), scorer=model_graded_fact())


# ---------------------------------------------------------------------------
# 2. system_message — inject a system prompt before calling the model
# ---------------------------------------------------------------------------
@task
def with_system_message():
    return Task(
        dataset=SAMPLES,
        solver=[
            system_message("You are a concise science tutor. Keep answers short."),
            generate(),
        ],
        scorer=model_graded_fact(),
    )


# ---------------------------------------------------------------------------
# 3. prompt_template — rewrite the user message using a template
#    {prompt} is replaced with the original sample input
# ---------------------------------------------------------------------------
@task
def with_prompt_template():
    template = "Answer the following question in exactly one sentence.\n\nQuestion: {prompt}"
    return Task(
        dataset=SAMPLES,
        solver=[prompt_template(template), generate()],
        scorer=model_graded_fact(),
    )


# ---------------------------------------------------------------------------
# 4. chain_of_thought — adds "think step by step" scaffolding
# ---------------------------------------------------------------------------
@task
def with_chain_of_thought():
    return Task(
        dataset=SAMPLES,
        solver=chain_of_thought(),
        scorer=model_graded_fact(),
    )


# ---------------------------------------------------------------------------
# 5. Custom solver — using the @solver decorator
#    A solver is an async function: TaskState -> TaskState
# ---------------------------------------------------------------------------
@solver
def add_reminder(reminder: str = "Be brief."):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Append reminder text to the last user message
        state.messages[-1].content += f"\n\n({reminder})"
        return await generate(state)

    return solve


@task
def with_custom_solver():
    return Task(
        dataset=SAMPLES,
        solver=add_reminder("Answer in 10 words or fewer."),
        scorer=model_graded_fact(),
    )


if __name__ == "__main__":
    tasks = [plain_generate, with_system_message, with_prompt_template, with_chain_of_thought, with_custom_solver]
    for t in tasks:
        results = eval(t(), model="mockllm/model", display="none")
        r = results[0]
        print(f"{r.task}: {r.results.scores[0].metrics}")
