"""Extract tool call requests from Anthropic and OpenAI LLM API responses.

Both Anthropic and OpenAI embed tool call requests inside their response
payloads, but in different shapes.  This module normalises both into a single
:class:`ToolCallRequest` dataclass.

Anthropic shape (content blocks)::

    {
        "content": [
            {"type": "text", "text": "I will search for that."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "search",
                "input": {"query": "hello"}
            }
        ]
    }

OpenAI shape (choices[0].message.tool_calls)::

    {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\": \"hello\"}"
                    }
                }]
            }
        }]
    }

Example::

    from llm_tool_call_extractor import ToolCallExtractor, ResponseFormat

    extractor = ToolCallExtractor()

    anthropic_response = {
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "search",
             "input": {"query": "hello"}}
        ]
    }
    calls = extractor.extract(anthropic_response)
    print(calls[0].name)        # "search"
    print(calls[0].arguments)   # {"query": "hello"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseFormat(str, Enum):
    """Known LLM response formats."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AUTO = "auto"


@dataclass
class ToolCallRequest:
    """A single normalised tool call request extracted from an LLM response.

    Attributes:
        id:         Provider-assigned call ID (empty string when not present).
        name:       Tool/function name.
        arguments:  Parsed arguments dict.  If the provider sent raw JSON, it
                    is parsed; malformed JSON results in an empty dict and the
                    raw string is stored in *raw_arguments*.
        raw_arguments: Original unparsed arguments string, or ``None`` when
                       the provider sent a dict directly.
        metadata:   Arbitrary extra fields from the provider block.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_argument(self, key: str) -> bool:
        """Return ``True`` if *key* is present in :attr:`arguments`."""
        return key in self.arguments

    def get_argument(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* from :attr:`arguments`, or *default*."""
        return self.arguments.get(key, default)


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Parse tool arguments into a dict.

    Returns ``(parsed_dict, raw_string_or_None)``.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return {}, None
    raw_str = str(raw)
    try:
        parsed = json.loads(raw_str)
        if not isinstance(parsed, dict):
            return {}, raw_str
        return parsed, raw_str
    except (json.JSONDecodeError, ValueError):
        return {}, raw_str


def _extract_anthropic(response: dict[str, Any]) -> list[ToolCallRequest]:
    """Extract tool_use blocks from an Anthropic-style response dict."""
    content = response.get("content", [])
    if not isinstance(content, list):
        return []

    calls: list[ToolCallRequest] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        call_id = block.get("id", "")
        raw_input = block.get("input")
        args, raw = _parse_arguments(raw_input)
        meta = {
            k: v for k, v in block.items() if k not in ("type", "id", "name", "input")
        }
        calls.append(
            ToolCallRequest(
                id=call_id,
                name=name,
                arguments=args,
                raw_arguments=raw,
                metadata=meta,
            )
        )
    return calls


def _extract_openai(response: dict[str, Any]) -> list[ToolCallRequest]:
    """Extract tool_calls from an OpenAI-style response dict."""
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return []

    # Use the first choice
    choice = choices[0]
    if not isinstance(choice, dict):
        return []

    message = choice.get("message", {})
    if not isinstance(message, dict):
        return []

    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []

    calls: list[ToolCallRequest] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        call_id = tc.get("id", "")
        func = tc.get("function", {})
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        raw_args = func.get("arguments")
        args, raw = _parse_arguments(raw_args)
        meta = {k: v for k, v in tc.items() if k not in ("id", "function")}
        calls.append(
            ToolCallRequest(
                id=call_id,
                name=name,
                arguments=args,
                raw_arguments=raw,
                metadata=meta,
            )
        )
    return calls


def _detect_format(response: dict[str, Any]) -> ResponseFormat:
    """Heuristically detect whether *response* is Anthropic or OpenAI."""
    if "choices" in response:
        return ResponseFormat.OPENAI
    if "content" in response:
        return ResponseFormat.ANTHROPIC
    # Fallback: inspect usage keys if present
    if "usage" in response:
        usage = response.get("usage", {})
        if "input_tokens" in usage:
            return ResponseFormat.ANTHROPIC
        if "prompt_tokens" in usage:
            return ResponseFormat.OPENAI
    return ResponseFormat.ANTHROPIC


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_from_anthropic(response: dict[str, Any]) -> list[ToolCallRequest]:
    """Extract tool calls from an Anthropic API response dict."""
    return _extract_anthropic(response)


def extract_from_openai(response: dict[str, Any]) -> list[ToolCallRequest]:
    """Extract tool calls from an OpenAI API response dict."""
    return _extract_openai(response)


class ToolCallExtractor:
    """Extract tool call requests from LLM API responses.

    Args:
        fmt: :class:`ResponseFormat` hint.  Use ``AUTO`` (default) to
             detect the format automatically.
    """

    def __init__(self, fmt: ResponseFormat = ResponseFormat.AUTO) -> None:
        self._fmt = fmt

    def extract(self, response: dict[str, Any]) -> list[ToolCallRequest]:
        """Extract all tool calls from *response*.

        The format is determined by the *fmt* parameter passed at
        construction, or auto-detected when ``fmt=AUTO``.

        Args:
            response: Raw API response dict.

        Returns:
            List of :class:`ToolCallRequest` (may be empty).
        """
        fmt = self._fmt
        if fmt == ResponseFormat.AUTO:
            fmt = _detect_format(response)

        if fmt == ResponseFormat.ANTHROPIC:
            return _extract_anthropic(response)
        if fmt == ResponseFormat.OPENAI:
            return _extract_openai(response)
        return []

    def has_tool_calls(self, response: dict[str, Any]) -> bool:
        """Return ``True`` if *response* contains at least one tool call."""
        return len(self.extract(response)) > 0

    def __repr__(self) -> str:
        return f"ToolCallExtractor(fmt={self._fmt.value!r})"
