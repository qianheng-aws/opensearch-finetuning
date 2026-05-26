from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from bedrock_judge import (
    JudgeError,
    build_prompt,
    judge_pairs,
    parse_score,
)


def test_build_prompt_contains_query_and_doc():
    p = build_prompt("python language", "Python is a language.")
    assert "python language" in p
    assert "Python is a language." in p
    assert "0.0" in p and "1.0" in p
    assert "single number" in p.lower()


def test_parse_score_plain_float():
    assert parse_score("0.7") == 0.7


def test_parse_score_strips_text():
    assert parse_score("Score: 0.85") == 0.85


def test_parse_score_clamps_above_one():
    assert parse_score("1.5") == 1.0


def test_parse_score_clamps_below_zero():
    assert parse_score("-0.2") == 0.0


def test_parse_score_raises_on_no_number():
    with pytest.raises(JudgeError, match="no numeric"):
        parse_score("very relevant!")


def _resp(text: str) -> dict:
    body = json.dumps({"content": [{"type": "text", "text": text}]})
    return {"body": MagicMock(read=lambda: body.encode())}


def test_judge_pairs_returns_score_per_pair():
    client = MagicMock()
    client.invoke_model.side_effect = [_resp("0.9"), _resp("0.1")]
    pairs = [
        {"query": "q1", "doc_id": "d1", "doc_text": "t1"},
        {"query": "q2", "doc_id": "d2", "doc_text": "t2"},
    ]
    out = judge_pairs(client, "model-x", pairs, concurrency=1)
    assert out == {("q1", "d1"): 0.9, ("q2", "d2"): 0.1}
    assert client.invoke_model.call_count == 2


def test_judge_pairs_uses_supplied_model_id():
    client = MagicMock()
    client.invoke_model.return_value = _resp("0.5")
    judge_pairs(client, "claude-haiku-4-5", [
        {"query": "q", "doc_id": "d", "doc_text": "t"},
    ], concurrency=1)
    assert client.invoke_model.call_args.kwargs["modelId"] == "claude-haiku-4-5"


def test_judge_pairs_dedups():
    client = MagicMock()
    client.invoke_model.return_value = _resp("0.5")
    pairs = [
        {"query": "q1", "doc_id": "d1", "doc_text": "t1"},
        {"query": "q1", "doc_id": "d1", "doc_text": "t1"},
    ]
    out = judge_pairs(client, "m", pairs, concurrency=1)
    assert client.invoke_model.call_count == 1
    assert out == {("q1", "d1"): 0.5}


def test_judge_pairs_retries_on_throttling(monkeypatch):
    from botocore.exceptions import ClientError
    monkeypatch.setattr("bedrock_judge.time.sleep", lambda s: None)
    client = MagicMock()
    err = ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
    client.invoke_model.side_effect = [err, _resp("0.7")]
    out = judge_pairs(client, "m", [
        {"query": "q", "doc_id": "d", "doc_text": "t"},
    ], concurrency=1, max_retries=3)
    assert out == {("q", "d"): 0.7}
    assert client.invoke_model.call_count == 2


def test_judge_pairs_gives_up_after_max_retries(monkeypatch):
    from botocore.exceptions import ClientError
    monkeypatch.setattr("bedrock_judge.time.sleep", lambda s: None)
    client = MagicMock()
    err = ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
    client.invoke_model.side_effect = err
    with pytest.raises(JudgeError, match="exhausted"):
        judge_pairs(client, "m", [
            {"query": "q", "doc_id": "d", "doc_text": "t"},
        ], concurrency=1, max_retries=2)
