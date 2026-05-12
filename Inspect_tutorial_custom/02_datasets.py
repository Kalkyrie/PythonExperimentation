"""
Tutorial 02 - Datasets
=======================
Three ways to supply data to an Inspect task:

  1. Inline list of Sample objects  (simplest)
  2. CSV file via csv_dataset()
  3. JSON file via json_dataset()

Run:
    inspect eval 02_datasets.py --model mockllm/model
    inspect eval 02_datasets.py@inline_dataset   --model anthropic/claude-haiku-4-5-20251001
    inspect eval 02_datasets.py@from_csv_dataset --model anthropic/claude-haiku-4-5-20251001
    inspect eval 02_datasets.py@from_json_dataset --model anthropic/claude-haiku-4-5-20251001
"""

import os
import csv
import json
import tempfile

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, csv_dataset, json_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


# ---------------------------------------------------------------------------
# 1. Inline dataset
# ---------------------------------------------------------------------------
@task
def inline_dataset():
    samples = [
        Sample(input="What is 2 + 2?",  target="4"),
        Sample(input="What is 10 - 3?", target="7"),
        Sample(input="What is 3 * 4?",  target="12"),
    ]
    return Task(dataset=samples, solver=generate(), scorer=exact())


# ---------------------------------------------------------------------------
# 2. CSV dataset
#
# CSV must have at minimum an "input" column.
# Optional columns: target, id, metadata (JSON string), sandbox
# ---------------------------------------------------------------------------
def _write_temp_csv():
    path = os.path.join(tempfile.gettempdir(), "inspect_demo.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target"])
        writer.writeheader()
        writer.writerows([
            {"input": "Capital of France?",  "target": "Paris"},
            {"input": "Capital of Germany?", "target": "Berlin"},
            {"input": "Capital of Japan?",   "target": "Tokyo"},
        ])
    return path


@task
def from_csv_dataset():
    csv_path = _write_temp_csv()
    return Task(dataset=csv_dataset(csv_path), solver=generate(), scorer=exact())


# ---------------------------------------------------------------------------
# 3. JSON dataset
#
# JSON must be a list of objects, each with an "input" key.
# ---------------------------------------------------------------------------
def _write_temp_json():
    path = os.path.join(tempfile.gettempdir(), "inspect_demo.json")
    samples = [
        {"input": "What colour is the sky?",  "target": "blue"},
        {"input": "What colour is grass?",    "target": "green"},
        {"input": "What colour is the sun?",  "target": "yellow"},
    ]
    with open(path, "w") as f:
        json.dump(samples, f)
    return path


@task
def from_json_dataset():
    json_path = _write_temp_json()
    return Task(dataset=json_dataset(json_path), solver=generate(), scorer=exact())


if __name__ == "__main__":
    for t in [inline_dataset, from_csv_dataset, from_json_dataset]:
        results = eval(t(), model="mockllm/model", display="none")
        r = results[0]
        print(f"{r.task}: {r.results.scores[0].metrics}")
