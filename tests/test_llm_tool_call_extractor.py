"""Tests for llm_tool_call_extractor."""

from __future__ import annotations

import json

from llm_tool_call_extractor import ResponseFormat, ToolCallExtractor, ToolCallRequest
from llm_tool_call_extractor.core import (
    _detect_format,
    extract_from_anthropic,
    extract_from_openai,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ANTHROPIC_SINGLE = {
    "content": [
        {"type": "text", "text": "I will search."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "search",
            "input": {"query": "hello", "limit": 5},
        },
    ]
}

ANTHROPIC_MULTI = {
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "search",
            "input": {"query": "hello"},
        },
        {
            "type": "tool_use",
            "id": "toolu_02",
            "name": "read_file",
            "input": {"path": "/tmp/x"},
        },
    ]
}

ANTHROPIC_NO_TOOLS = {
    "content": [
        {"type": "text", "text": "Hello!"},
    ]
}

OPENAI_SINGLE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "hello"}),
                        },
                    }
                ],
            }
        }
    ]
}

OPENAI_MULTI = {
    "choices": [
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"q": "a"}),
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"path": "/"}),
                        },
                    },
                ]
            }
        }
    ]
}

OPENAI_NO_TOOLS = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}


# ---------------------------------------------------------------------------
# ToolCallRequest
# ---------------------------------------------------------------------------


def test_request_defaults():
    r = ToolCallRequest(id="x", name="search")
    assert r.arguments == {}
    assert r.raw_arguments is None
    assert r.metadata == {}


def test_request_has_argument():
    r = ToolCallRequest(id="x", name="t", arguments={"q": "hello"})
    assert r.has_argument("q")
    assert not r.has_argument("missing")


def test_request_get_argument():
    r = ToolCallRequest(id="x", name="t", arguments={"q": "hello"})
    assert r.get_argument("q") == "hello"
    assert r.get_argument("missing") is None
    assert r.get_argument("missing", "default") == "default"


# ---------------------------------------------------------------------------
# extract_from_anthropic
# ---------------------------------------------------------------------------


def test_anthropic_single():
    calls = extract_from_anthropic(ANTHROPIC_SINGLE)
    assert len(calls) == 1
    c = calls[0]
    assert c.id == "toolu_01"
    assert c.name == "search"
    assert c.arguments == {"query": "hello", "limit": 5}


def test_anthropic_multi():
    calls = extract_from_anthropic(ANTHROPIC_MULTI)
    assert len(calls) == 2
    assert calls[0].name == "search"
    assert calls[1].name == "read_file"


def test_anthropic_no_tools():
    calls = extract_from_anthropic(ANTHROPIC_NO_TOOLS)
    assert calls == []


def test_anthropic_empty_content():
    calls = extract_from_anthropic({"content": []})
    assert calls == []


def test_anthropic_missing_content():
    calls = extract_from_anthropic({})
    assert calls == []


def test_anthropic_text_blocks_skipped():
    calls = extract_from_anthropic(ANTHROPIC_SINGLE)
    names = [c.name for c in calls]
    assert "text" not in names


def test_anthropic_raw_arguments_none_for_dict():
    calls = extract_from_anthropic(ANTHROPIC_SINGLE)
    assert calls[0].raw_arguments is None


def test_anthropic_no_id():
    resp = {"content": [{"type": "tool_use", "name": "ping", "input": {}}]}
    calls = extract_from_anthropic(resp)
    assert calls[0].id == ""


def test_anthropic_no_name():
    resp = {"content": [{"type": "tool_use", "id": "x", "input": {}}]}
    calls = extract_from_anthropic(resp)
    assert calls[0].name == ""


def test_anthropic_extra_fields_in_metadata():
    resp = {
        "content": [
            {
                "type": "tool_use",
                "id": "x",
                "name": "t",
                "input": {},
                "custom_field": "value",
            }
        ]
    }
    calls = extract_from_anthropic(resp)
    assert calls[0].metadata.get("custom_field") == "value"


def test_anthropic_null_input():
    resp = {"content": [{"type": "tool_use", "id": "x", "name": "t", "input": None}]}
    calls = extract_from_anthropic(resp)
    assert calls[0].arguments == {}


# ---------------------------------------------------------------------------
# extract_from_openai
# ---------------------------------------------------------------------------


def test_openai_single():
    calls = extract_from_openai(OPENAI_SINGLE)
    assert len(calls) == 1
    c = calls[0]
    assert c.id == "call_abc"
    assert c.name == "search"
    assert c.arguments == {"query": "hello"}


def test_openai_multi():
    calls = extract_from_openai(OPENAI_MULTI)
    assert len(calls) == 2
    assert calls[0].name == "search"
    assert calls[1].name == "read"


def test_openai_no_tools():
    calls = extract_from_openai(OPENAI_NO_TOOLS)
    assert calls == []


def test_openai_empty_choices():
    calls = extract_from_openai({"choices": []})
    assert calls == []


def test_openai_missing_choices():
    calls = extract_from_openai({})
    assert calls == []


def test_openai_raw_arguments_stored():
    calls = extract_from_openai(OPENAI_SINGLE)
    # raw_arguments is the JSON string
    raw = calls[0].raw_arguments
    assert raw is not None
    assert json.loads(raw) == {"query": "hello"}


def test_openai_malformed_json_args():
    bad = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "t", "arguments": "not json {"},
                        }
                    ]
                }
            }
        ]
    }
    calls = extract_from_openai(bad)
    assert calls[0].arguments == {}
    assert calls[0].raw_arguments == "not json {"


def test_openai_type_in_metadata():
    calls = extract_from_openai(OPENAI_SINGLE)
    assert calls[0].metadata.get("type") == "function"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_detect_anthropic():
    assert _detect_format(ANTHROPIC_SINGLE) == ResponseFormat.ANTHROPIC


def test_detect_openai():
    assert _detect_format(OPENAI_SINGLE) == ResponseFormat.OPENAI


def test_detect_empty_fallback():
    # No strong signal — falls back to ANTHROPIC
    assert _detect_format({}) == ResponseFormat.ANTHROPIC


def test_detect_usage_tokens_anthropic():
    resp = {"usage": {"input_tokens": 10, "output_tokens": 5}}
    assert _detect_format(resp) == ResponseFormat.ANTHROPIC


def test_detect_usage_tokens_openai():
    resp = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert _detect_format(resp) == ResponseFormat.OPENAI


# ---------------------------------------------------------------------------
# ToolCallExtractor
# ---------------------------------------------------------------------------


def test_extractor_auto_anthropic():
    ex = ToolCallExtractor()
    calls = ex.extract(ANTHROPIC_SINGLE)
    assert len(calls) == 1
    assert calls[0].name == "search"


def test_extractor_auto_openai():
    ex = ToolCallExtractor()
    calls = ex.extract(OPENAI_SINGLE)
    assert len(calls) == 1
    assert calls[0].name == "search"


def test_extractor_explicit_anthropic():
    ex = ToolCallExtractor(fmt=ResponseFormat.ANTHROPIC)
    calls = ex.extract(ANTHROPIC_SINGLE)
    assert len(calls) == 1


def test_extractor_explicit_openai():
    ex = ToolCallExtractor(fmt=ResponseFormat.OPENAI)
    calls = ex.extract(OPENAI_SINGLE)
    assert len(calls) == 1


def test_extractor_has_tool_calls_true():
    ex = ToolCallExtractor()
    assert ex.has_tool_calls(ANTHROPIC_SINGLE) is True


def test_extractor_has_tool_calls_false():
    ex = ToolCallExtractor()
    assert ex.has_tool_calls(ANTHROPIC_NO_TOOLS) is False


def test_extractor_has_tool_calls_openai():
    ex = ToolCallExtractor()
    assert ex.has_tool_calls(OPENAI_SINGLE) is True
    assert ex.has_tool_calls(OPENAI_NO_TOOLS) is False


def test_extractor_repr():
    ex = ToolCallExtractor()
    assert "auto" in repr(ex)


def test_extractor_anthropic_multi():
    ex = ToolCallExtractor()
    calls = ex.extract(ANTHROPIC_MULTI)
    assert len(calls) == 2


def test_extractor_openai_multi():
    ex = ToolCallExtractor()
    calls = ex.extract(OPENAI_MULTI)
    assert len(calls) == 2


def test_extractor_no_calls_returns_empty():
    ex = ToolCallExtractor()
    assert ex.extract({}) == []


def test_extractor_arguments_are_dict():
    ex = ToolCallExtractor()
    calls = ex.extract(ANTHROPIC_SINGLE)
    assert isinstance(calls[0].arguments, dict)


def test_openai_dict_arguments_parsed():
    # Some implementations may pass dict directly instead of JSON string
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "t", "arguments": {"key": "val"}},
                        }
                    ]
                }
            }
        ]
    }
    calls = extract_from_openai(resp)
    assert calls[0].arguments == {"key": "val"}
    assert calls[0].raw_arguments is None


def test_empty_response_has_no_tool_calls():
    ex = ToolCallExtractor()
    assert not ex.has_tool_calls({})


def test_request_metadata_not_polluted():
    calls = extract_from_anthropic(ANTHROPIC_SINGLE)
    # type, id, name, input should NOT be in metadata
    meta = calls[0].metadata
    assert "type" not in meta
    assert "id" not in meta
    assert "name" not in meta
    assert "input" not in meta


def test_extractor_format_enum_values():
    assert ResponseFormat.ANTHROPIC.value == "anthropic"
    assert ResponseFormat.OPENAI.value == "openai"
    assert ResponseFormat.AUTO.value == "auto"


def test_extractor_explicit_auto():
    ex = ToolCallExtractor(fmt=ResponseFormat.AUTO)
    assert ex.has_tool_calls(ANTHROPIC_SINGLE) is True
    assert ex.has_tool_calls(OPENAI_SINGLE) is True


def test_anthropic_content_not_list():
    calls = extract_from_anthropic({"content": "just a string"})
    assert calls == []


def test_openai_first_choice_used():
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "first",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        }
                    ]
                }
            },
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "second",
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        }
                    ]
                }
            },
        ]
    }
    calls = extract_from_openai(resp)
    assert len(calls) == 1
    assert calls[0].id == "first"


def test_request_get_argument_none_value():
    # None is a valid argument value, distinguish from missing
    r = ToolCallRequest(id="x", name="t", arguments={"key": None})
    assert r.has_argument("key") is True
    assert r.get_argument("key") is None


def test_extractor_unknown_fmt_returns_empty():
    # Directly inject an unknown value by bypassing the enum (edge case)
    ex = ToolCallExtractor.__new__(ToolCallExtractor)
    ex._fmt = "unknown_format"  # type: ignore[assignment]
    # Unknown fmt falls through all conditionals → returns empty list
    result = ex.extract(ANTHROPIC_SINGLE)
    assert result == []


def test_openai_empty_string_arguments():
    # OpenAI commonly sends "" for no-argument tool calls.
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "now", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }
    calls = extract_from_openai(resp)
    assert calls[0].arguments == {}
    assert calls[0].raw_arguments == ""


def test_openai_missing_arguments_field():
    # A tool call with no "arguments" key at all yields an empty dict.
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "now"},
                        }
                    ]
                }
            }
        ]
    }
    calls = extract_from_openai(resp)
    assert calls[0].arguments == {}
    assert calls[0].raw_arguments is None


def test_helpers_importable_from_package_root():
    # extract_from_anthropic / extract_from_openai are part of the public API.
    from llm_tool_call_extractor import (
        extract_from_anthropic as pkg_anthropic,
    )
    from llm_tool_call_extractor import (
        extract_from_openai as pkg_openai,
    )

    assert pkg_anthropic is extract_from_anthropic
    assert pkg_openai is extract_from_openai
