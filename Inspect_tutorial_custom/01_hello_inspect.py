"""
Tutorial 01 - Hello Inspect
============================
The smallest possible Inspect evaluation.

Concepts introduced:
  - @task decorator
  - Task, Sample, generate(), exact()
  - Running with inspect.eval() in Python
  - Running from the CLI

Run (CLI):
    inspect eval 01_hello_inspect.py --model anthropic/claude-haiku-4-5-20251001

Run (Python):
    python3 01_hello_inspect.py

Run offline (no API key, uses mock model):
    inspect eval 01_hello_inspect.py --model mockllm/model
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_inspect():
    return Task(
        dataset=[
            Sample(input="Reply with exactly: Hello World", target="Hello World"),
            Sample(input="Reply with exactly: Inspect AI", target="Inspect AI"),
        ],
        solver=generate(),
        scorer=exact(),
    )


if __name__ == "__main__":
    # Replace model string with your provider, e.g. "anthropic/claude-haiku-4-5-20251001"
    results = eval(hello_inspect(), model="mockllm/model", display="none")
    for result in results:
        print(f"Task : {result.task}")
        print(f"Model: {result.model}")
        print(f"Score: {result.results.scores[0].metrics}")
