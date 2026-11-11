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
        task.messages.append({"role": "user", "content": f"Output is not valid JSON: {exc}"})
        return task


client = AsyncOpenAI(api_key="your-api-key")

runner = Runner(
    client=client,
    max_concurrency=128,
    max_retries=2,
    retry_delay=1.0,
)

messages_list = [
    [{"role": "user", "content": "Return JSON: {\"name\": \"Ada\"}"}],
    [{"role": "user", "content": "Return JSON: {\"name\": \"Grace\"}"}],
]

# create tasks with any extra attributes you want to use in the pipeline or final results
# here we attach a `customer` dict as the extra attribute
tasks = [
    Task(messages=messages_list[0], customer={"name": "Ada"}),
    Task(messages=messages_list[1], customer={"name": "Grace"}),
]

results = runner.run(
    tasks,
    pipeline=[parse_json],
    model="gpt-4o-mini",
    temperature=0.2,
    stream=True,
    interactive=True,
)

for result in results:
    print(result.index, result.status, result.output, result.error, result.messages, result.customer)
```

Each result is a `Task` with:

- `index`: input position
- `messages`: current message list, reused for follow-up turns
  and including the latest assistant reply from the backend
- any extra attributes you attached when creating the task, such as `customer`
- `status`: `success`, `error`, or `cancelled`
- `output`: latest model text, or the final transformed value returned by the pipeline
- `error`: request or parse error string
- `attempts`: attempts used by the most recent backend turn
- `metadata`: lifecycle metadata including `retry_count`, `total_backend_calls`, and `last_text`

Pipeline stages accept a `Task` and may return:

- a `Task` to enqueue another backend request with the updated `task.messages`
- any non-`Task` value to finish that slot successfully

After every new backend response, the pipeline restarts from the first stage.

## Recovery After Interruptions

`Runner` keeps the latest batch state in `runner.last_results`.

```python
try:
    results = runner.run(tasks, pipeline=[parse_json], model="gpt-4o-mini", temperature=0.2)
except KeyboardInterrupt:
    results = runner.last_results
```

Common request settings such as `model`, `temperature`, `max_tokens`, `top_p`, `stop`,
`presence_penalty`, `frequency_penalty`, and `timeout` are exposed directly on `run()` and `arun()`.
Less common OpenAI request options can still be passed through as extra keyword arguments.

### Per-Task Backend Options

Use `backend_kwargs` to override batch request settings for an individual task. It defaults
to an independent empty dictionary. `client_kwarg()` builds this dictionary from keyword arguments:

```python
from llmrunner import Task, client_kwarg

task = Task(
    messages=[{"role": "user", "content": "Return a JSON object."}],
    customer_id="customer-1",
    backend_kwargs=client_kwarg(
        response_format={"type": "json_object"},
        max_tokens=512,
    ),
)
results = await runner.arun([task], model="gpt-4o-mini", max_tokens=1024)
```

This task uses `max_tokens=512`; `customer_id` remains local metadata and is not sent to
the client. A plain dictionary also works. Options are applied on every request, including
retries and pipeline follow-up turns. Pipelines returning a new `Task` should pass
`backend_kwargs=task.backend_kwargs` to retain the previous task's options.
`messages` and `stream` cannot be set inside `backend_kwargs` because the runner controls them.
The effective request must specify `model`, either in the batch settings or in `backend_kwargs`.

## Streaming and interactive dashboards

`run()` and `arun()` stream Chat Completions and show an interactive dashboard by default:

```python
results = runner.run(tasks, model="gpt-4o-mini", stream=True, interactive=True)
```

In Jupyter, use the asynchronous API. The dashboard uses native async widget-event handling
on ipykernel 7.4+ and polls UI events on older kernels, so paging, sorting, and expansion
remain responsive during a batch:

```python
results = await runner.arun(tasks, model="gpt-4o-mini")
```

The dashboard displays 10 tasks per page with sortable columns for task index, generated
character count, and completion. Expanding a task shows its complete message history plus the
current assistant draft. The detail area preserves line breaks and scrolls when the content is
long. Updates are batched to at most once every 100 ms, and the controls are updated in place so
streaming refreshes do not replace a button while it is being clicked.

In a terminal, the same data is shown in an inline Textual application. Select a row and press
Enter to expand it, click a column header to sort, use Left/Right or the page buttons to change
pages, and press `q` to hide the dashboard without cancelling the batch.

The display behavior is controlled independently from backend streaming:

| `interactive` | `show_progress` | Display |
| --- | --- | --- |
| `True` | `True` | Interactive table and progress bar |
| `True` | `False` | Interactive table only |
| `False` | `True` | tqdm progress bar only |
| `False` | `False` | No display |

Set `stream=False` to request a regular non-streaming response. The interactive table remains
available, but its generated message and character count update only after each response is
complete. `run()` and `arun()` always return the final ordered `list[Task]`; they do not return a
stream iterator.

The character column counts successfully completed assistant text across all pipeline turns plus
the current draft. Partial output from a failed attempt is discarded before retrying. Progress
treats `success`, `error`, and `cancelled` as completed terminal states and reports completed,
remaining, and estimated time below the table.

## Client Support

`Runner` currently supports official OpenAI Python clients exposing `client.chat.completions.create(...)`.

- `AsyncOpenAI` is the fast path
- `OpenAI` also works and is wrapped for async execution internally
- unsupported client types raise `NotImplementedError`

Completed items remain available, and unfinished items are marked as `cancelled`.

While a batch is still running, unfinished slots may have `status=None` until they reach a final outcome.
