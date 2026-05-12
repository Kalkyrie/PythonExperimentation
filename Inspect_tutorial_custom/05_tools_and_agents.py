"""
Tutorial 05 - Tools and Agents
================================
Inspect supports agentic evaluations where the model can call tools
in a loop until it decides it's done.

This tutorial covers:
  1. Defining a custom tool with @tool
  2. Using basic_agent() for an autonomous tool-calling loop
  3. Controlling the agent loop (max_attempts, message_limit)

Note: tool-calling requires a real model that supports it.
      With mockllm the agent loop will stop after one turn.

Run with a real model:
    inspect eval 05_tools_and_agents.py --model anthropic/claude-haiku-4-5-20251001

Run offline (limited behaviour):
    inspect eval 05_tools_and_agents.py --model mockllm/model
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent, generate, system_message
from inspect_ai.tool import tool


# ---------------------------------------------------------------------------
# Custom tool — annotate with @tool so Inspect can serialise it for the LLM
# ---------------------------------------------------------------------------
@tool
def calculator():
    async def execute(expression: str) -> str:
        """
        Evaluate a simple arithmetic expression.

        Args:
            expression: A Python arithmetic expression, e.g. "2 + 2" or "100 / 4"
        """
        try:
            result = eval(compile(expression, "<string>", "eval"), {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return execute


@tool
def word_counter():
    async def execute(text: str) -> str:
        """
        Count the number of words in a text string.

        Args:
            text: The text to count words in.
        """
        count = len(text.split())
        return f"{count} words"

    return execute


# ---------------------------------------------------------------------------
# Task: give the agent a problem that requires tool use
# ---------------------------------------------------------------------------
@task
def agent_with_tools():
    return Task(
        dataset=[
            Sample(
                input="Use the calculator to compute 123 * 456, then tell me the result.",
                target="56088",
            ),
            Sample(
                input="Count the words in: 'The quick brown fox jumps over the lazy dog'",
                target="9",
            ),
        ],
        solver=basic_agent(
            tools=[calculator(), word_counter()],
            message_limit=10,
        ),
        scorer=includes(),
    )


# ---------------------------------------------------------------------------
# Variant: add a system prompt to guide the agent
# ---------------------------------------------------------------------------
@task
def guided_agent():
    return Task(
        dataset=[
            Sample(
                input="What is 17 * 23? Use the calculator tool.",
                target="391",
            ),
        ],
        solver=[
            system_message("You are a precise assistant. Always use the provided tools."),
            basic_agent(tools=[calculator()], message_limit=6),
        ],
        scorer=includes(),
    )


if __name__ == "__main__":
    for t in [agent_with_tools, guided_agent]:
        results = eval(t(), model="mockllm/model", display="none")
        r = results[0]
        print(f"{r.task}: {r.results.scores[0].metrics}")
