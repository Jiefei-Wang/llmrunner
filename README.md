# aclient

Importable async LLM client package with ordered batch execution, retries, progress reporting, and recoverable partial results.

## Install

```bash
pip install .
```

With OpenAI support:

```bash
pip install .[openai]
```

## Basic Usage

```python
import json

from aclient import Client, OpenAIBackend


def parse_json(text: str):
    return json.loads(text)


backend = OpenAIBackend(
    model="gpt-4o-mini",
    api_key="your-api-key",
)

client = Client(
    backend=backend,
    max_concurrency=128,
    max_retries=2,
    retry_delay=1.0,
)

histories = [
    [{"role": "user", "content": "Return JSON: {\"name\": \"Ada\"}"}],
    [{"role": "user", "content": "Return JSON: {\"name\": \"Grace\"}"}],
]

results = client.run(histories, parse_fn=parse_json)

for result in results:
    print(result.index, result.status, result.parsed, result.error)
```

Each result is a `ResultRecord` with:

- `index`: input position
- `messages`: original request messages
- `status`: `success`, `error`, or `cancelled`
- `text`: raw model response text
- `parsed`: parsed output from `parse_fn(text)` if provided
- `error`: request or parse error string

## Recovery After Interruptions

`Client` keeps the latest batch state in `client.last_results`.

```python
try:
    results = client.run(histories, parse_fn=parse_json)
except KeyboardInterrupt:
    results = client.last_results
```

Completed items remain available, and unfinished items are marked as `cancelled`.

While a batch is still running, unfinished slots may have `status=None` until they reach a final outcome.
