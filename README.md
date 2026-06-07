# llm-tool-call-extractor

Extract structured tool call requests from Anthropic and OpenAI API responses. Zero dependencies.

## Install

```bash
pip install llm-tool-call-extractor
```

## Usage

```python
from llm_tool_call_extractor import ToolCallExtractor

extractor = ToolCallExtractor()  # auto-detects format

# Anthropic response
anthropic_response = {
    "content": [
        {"type": "text", "text": "I will search for that."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "search",
            "input": {"query": "hello", "limit": 5}
        }
    ]
}
calls = extractor.extract(anthropic_response)
print(calls[0].name)        # "search"
print(calls[0].arguments)   # {"query": "hello", "limit": 5}
print(calls[0].id)          # "toolu_01"

# OpenAI response
import json
openai_response = {
    "choices": [{
        "message": {
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "hello"})
                }
            }]
        }
    }]
}
calls = extractor.extract(openai_response)
print(calls[0].name)  # "search"

# Check if any tool calls exist
if extractor.has_tool_calls(response):
    for call in extractor.extract(response):
        result = dispatch(call.name, call.arguments)
```

## Explicit format

```python
from llm_tool_call_extractor import (
    ToolCallExtractor,
    ResponseFormat,
    extract_from_anthropic,
    extract_from_openai,
)

# Explicit format
ex = ToolCallExtractor(fmt=ResponseFormat.ANTHROPIC)
ex = ToolCallExtractor(fmt=ResponseFormat.OPENAI)

# Standalone helpers
calls = extract_from_anthropic(response)
calls = extract_from_openai(response)
```

## ToolCallRequest fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Provider-assigned call ID |
| `name` | `str` | Tool/function name |
| `arguments` | `dict` | Parsed arguments (JSON decoded if needed) |
| `raw_arguments` | `str \| None` | Original JSON string (OpenAI), or `None` (Anthropic dict) |
| `metadata` | `dict` | Extra provider fields |

```python
call.has_argument("query")         # True/False
call.get_argument("query", None)   # value or default
```

## License

MIT
