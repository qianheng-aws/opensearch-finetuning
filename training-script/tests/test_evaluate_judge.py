from __future__ import annotations

import io
import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from evaluate_judge import (
    _aws_region,
    build_judge_pairs,
    parse_args,
    main,
    _to_dynamodb_safe,
)


def test_aws_region_reads_AWS_REGION(monkeypatch):
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert _aws_region() == "us-east-1"


def test_aws_region_falls_back_to_AWS_DEFAULT_REGION(monkeypatch):
    """SageMaker training containers set AWS_DEFAULT_REGION but not AWS_REGION."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    assert _aws_region() == "us-west-2"


def test_aws_region_returns_None_when_neither_set(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    assert _aws_region() is None


def test_parse_args_required():
    args = parse_args([
        "--queries-s3", "s3://b/q",
        "--pool-s3", "s3://b/p",
        "--base-top10-s3", "s3://b/bt",
        "--ft-top10-s3", "s3://b/ft",
        "--ddb-table-name", "T",
        "--task-id", "tid",
        "--out-s3", "s3://b/o",
    ])
    assert args.judge_concurrency == 16
    assert args.top_k == 10
    assert args.bootstrap_resamples == 1000


def test_build_judge_pairs_unions_and_dedups():
    queries = [{"query": "q1", "doc_id": "d_a"}]
    base_top10 = [{"query": "q1", "top10": ["d1", "d2", "d3"]}]
    ft_top10 = [{"query": "q1", "top10": ["d2", "d4"]}]
    doc_text = {"d1": "t1", "d2": "t2", "d3": "t3", "d4": "t4"}
    pairs = build_judge_pairs(queries, base_top10, ft_top10, doc_text)
    keys = {(p["query"], p["doc_id"]) for p in pairs}
    assert keys == {("q1", d) for d in ("d1", "d2", "d3", "d4")}
    by_id = {p["doc_id"]: p for p in pairs}
    assert by_id["d1"]["doc_text"] == "t1"


def test_to_dynamodb_safe_converts_floats_to_decimal():
    payload = {"a": 1.5, "b": [0.1, 0.2], "c": {"d": 3.14}}
    out = _to_dynamodb_safe(payload)
    assert out["a"] == Decimal("1.5")
    assert out["b"][0] == Decimal("0.1")
    assert out["c"]["d"] == Decimal("3.14")


def _resp(text: str) -> dict:
    body = json.dumps({"content": [{"type": "text", "text": text}]})
    return {"body": MagicMock(read=lambda: body.encode())}


def test_main_writes_report_and_ddb_row(monkeypatch):
    s3 = MagicMock()

    queries_b = (json.dumps({"query": "q1", "doc_id": "d_a"}) + "\n"
                 + json.dumps({"query": "q2", "doc_id": "d_c"}) + "\n").encode()
    pool_b = "\n".join(
        json.dumps({"id": d, "text": d}) for d in ("d_a", "d_b", "d_c")
    ).encode()
    base_b = (json.dumps({"query": "q1", "top10": ["d_b", "d_a"]}) + "\n"
              + json.dumps({"query": "q2", "top10": ["d_b", "d_c"]}) + "\n").encode()
    ft_b = (json.dumps({"query": "q1", "top10": ["d_a", "d_b"]}) + "\n"
            + json.dumps({"query": "q2", "top10": ["d_c", "d_b"]}) + "\n").encode()

    def get_object(Bucket, Key):
        if Key.endswith("queries.jsonl"):
            return {"Body": io.BytesIO(queries_b)}
        if Key.endswith("pool.jsonl"):
            return {"Body": io.BytesIO(pool_b)}
        if Key.endswith("base.jsonl"):
            return {"Body": io.BytesIO(base_b)}
        if Key.endswith("ft.jsonl"):
            return {"Body": io.BytesIO(ft_b)}
        raise KeyError(Key)
    s3.get_object.side_effect = get_object
    s3.put_object = MagicMock()

    bedrock = MagicMock()
    def _judge(modelId, body, **kw):
        msg = json.loads(body)["messages"][0]["content"]
        # Heuristic: gold pairs (q1↔d_a, q2↔d_c) -> 0.9; else 0.1
        score = "0.1"
        try:
            doc_part = msg.split("Document:")[1]
        except IndexError:
            doc_part = ""
        if ("q1" in msg and "d_a" in doc_part) or ("q2" in msg and "d_c" in doc_part):
            score = "0.9"
        return _resp(score)
    bedrock.invoke_model.side_effect = _judge

    table = MagicMock()
    ddb = MagicMock()
    ddb.Table = MagicMock(return_value=table)

    monkeypatch.setattr("evaluate_judge._build_s3_client", lambda: s3)
    monkeypatch.setattr("evaluate_judge._build_bedrock_client", lambda: bedrock)
    monkeypatch.setattr("evaluate_judge._build_ddb_resource", lambda: ddb)

    main([
        "--queries-s3", "s3://b/queries.jsonl",
        "--pool-s3", "s3://b/pool.jsonl",
        "--base-top10-s3", "s3://b/base.jsonl",
        "--ft-top10-s3", "s3://b/ft.jsonl",
        "--judge-concurrency", "1",
        "--top-k", "2",
        "--bootstrap-resamples", "100",
        "--seed", "42",
        "--ddb-table-name", "EvalResults",
        "--task-id", "test-1",
        "--out-s3", "s3://b/evaluation_report.json",
    ])

    put_calls = [c for c in s3.put_object.call_args_list
                 if c.kwargs["Key"] == "evaluation_report.json"]
    assert len(put_calls) == 1
    report = json.loads(put_calls[0].kwargs["Body"].decode())
    assert report["task_id"] == "test-1"
    metric = report["metrics"]["ndcg_at_2"]
    assert metric["fine_tuned"] > metric["base"]
    assert metric["uplift_mean"] > 0

    ddb.Table.assert_called_once_with("EvalResults")
    item = table.put_item.call_args.kwargs["Item"]
    assert item["task_id"] == "test-1"
    assert "metrics" in item
