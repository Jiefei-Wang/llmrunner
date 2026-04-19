# llmrunner

Importable async LLM runner package with ordered batch execution, retries, progress reporting, and task-driven multi-turn pipelines.

## Install

```bash
pip install .
```

## Basic Usage

```python
import json

from openai import AsyncOpenAI

from llmrunner import Runner, Task


def parse_json(task: Task):
    output = task.output
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        task.history += [
            {"role": "assistant", "content": output},
            {"role": "user", "content": f"Output is not valid JSON: {exc}"},
        ]
        return task


client = AsyncOpenAI(api_key="your-api-key")

runner = Runner(
    client=client,
    max_concurrency=128,
    max_retries=2,
    retry_delay=1.0,
)

histories = [
    [{"role": "user", "content": "Return JSON: {\"name\": \"Ada\"}"}],
    [{"role": "user", "content": "Return JSON: {\"name\": \"Grace\"}"}],
]

results = runner.run(histories, pipeline=[parse_json], model="gpt-4o-mini")

for result in results:
    print(result.index, result.status, result.output, result.error, result.history)
```

Each result is a `Task` with:

- `index`: input position
- `history`: current message history, reused for follow-up turns
- `status`: `success`, `error`, or `cancelled`
- `output`: latest model text, or the final transformed value returned by the pipeline
- `error`: request or parse error string
- `attempts`: attempts used by the most recent backend turn
- `metadata`: lifecycle metadata including `retry_count`, `total_backend_calls`, and `last_text`

Pipeline stages accept a `Task` and may return:

- a `Task` to enqueue another backend request with the updated `task.history`
- any non-`Task` value to finish that slot successfully

After every new backend response, the pipeline restarts from the first stage.

## Recovery After Interruptions

`Runner` keeps the latest batch state in `runner.last_results`.

```python
try:
    results = runner.run(histories, pipeline=[parse_json], model="gpt-4o-mini")
except KeyboardInterrupt:
    results = runner.last_results
```

## Client Support

`Runner` currently supports official OpenAI Python clients exposing `client.chat.completions.create(...)`.

- `AsyncOpenAI` is the fast path
- `OpenAI` also works and is wrapped for async execution internally
- unsupported client types raise `NotImplementedError`

Completed items remain available, and unfinished items are marked as `cancelled`.

While a batch is still running, unfinished slots may have `status=None` until they reach a final outcome.
