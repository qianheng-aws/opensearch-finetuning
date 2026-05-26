from __future__ import annotations

import io
import json
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from index import (
    extract_eval_queries_from_tarball,
    sample_stratified_queries,
    build_kNN_query_body,
    parse_search_hits,
    handler,
)


def _make_tarball_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def test_extract_eval_queries_reads_jsonl_from_tarball():
    payload = (
        json.dumps({"query": "q1", "doc_id": "d1"}) + "\n"
        + json.dumps({"query": "q2", "doc_id": "d2"}) + "\n"
    ).encode()
    tarbytes = _make_tarball_bytes({"eval_queries.jsonl": payload, "training_data.jsonl": b""})
    result = extract_eval_queries_from_tarball(tarbytes)
    assert result == [
        {"query": "q1", "doc_id": "d1"},
        {"query": "q2", "doc_id": "d2"},
    ]


def test_extract_raises_when_eval_queries_missing():
    tarbytes = _make_tarball_bytes({"training_data.jsonl": b""})
    with pytest.raises(ValueError, match="eval_queries.jsonl"):
        extract_eval_queries_from_tarball(tarbytes)


def test_sample_stratified_takes_balanced_per_doc():
    pool = []
    for d in range(5):
        for q in range(4):
            pool.append({"query": f"d{d}_q{q}", "doc_id": f"d{d}"})
    sampled = sample_stratified_queries(pool, num=10, seed=42)
    assert len(sampled) == 10
    counts = {}
    for q in sampled:
        counts[q["doc_id"]] = counts.get(q["doc_id"], 0) + 1
    assert all(c == 2 for c in counts.values())


def test_sample_caps_at_pool_size():
    pool = [{"query": f"q{i}", "doc_id": f"d{i}"} for i in range(5)]
    assert len(sample_stratified_queries(pool, num=10, seed=1)) == 5


def test_sample_is_deterministic():
    pool = [{"query": f"q{i}", "doc_id": f"d{i % 3}"} for i in range(30)]
    a = sample_stratified_queries(pool, num=9, seed=7)
    b = sample_stratified_queries(pool, num=9, seed=7)
    assert [x["query"] for x in a] == [x["query"] for x in b]


def test_build_kNN_query_body_uses_neural_query_with_match_text():
    body = build_kNN_query_body("python language", text_field="text", size=100)
    assert body["size"] == 100
    query_str = json.dumps(body)
    assert "python language" in query_str


def test_parse_search_hits_extracts_id_and_text():
    response = {
        "hits": {"hits": [
            {"_id": "d1", "_source": {"text": "doc one"}},
            {"_id": "d2", "_source": {"text": "doc two"}},
        ]}
    }
    hits = parse_search_hits(response, text_field="text")
    assert hits == [
        {"id": "d1", "text": "doc one"},
        {"id": "d2", "text": "doc two"},
    ]


def test_parse_search_hits_skips_missing_text():
    response = {
        "hits": {"hits": [
            {"_id": "d1", "_source": {}},
            {"_id": "d2", "_source": {"text": "ok"}},
        ]}
    }
    hits = parse_search_hits(response, text_field="text")
    assert hits == [{"id": "d2", "text": "ok"}]


def test_handler_writes_three_s3_files_and_returns_uris(monkeypatch):
    payload = "\n".join(
        json.dumps({"query": f"q{i}", "doc_id": f"d{i % 3}"}) for i in range(15)
    ).encode()
    tarbytes = _make_tarball_bytes({"eval_queries.jsonl": payload})

    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(tarbytes)}
    s3.put_object = MagicMock()

    aoss_search = MagicMock()
    aoss_search.return_value = {
        "hits": {"hits": [
            {"_id": "doc_x", "_source": {"text": "x"}},
            {"_id": "doc_y", "_source": {"text": "y"}},
        ]}
    }

    monkeypatch.setattr("index._build_s3_client", lambda: s3)
    monkeypatch.setattr("index._aoss_search", aoss_search)

    event = {
        "task_id": "task-1",
        "model_name": "smoke",
        "opensearch_endpoint": "https://aoss.example.com",
        "opensearch_index_name": "test-idx",
        "eval_queries_s3": "s3://bucket/path/to/model.tar.gz",
        "data_bucket": "bucket",
        "data_prefix": "smoke/evaluation",
        "num_eval_queries": 5,
        "pool_size": 10,
        "top_k": 3,
        "seed": 42,
        "text_field": "text",
    }
    result = handler(event, None)

    assert result["queries_s3"].startswith("s3://bucket/smoke/evaluation/")
    assert result["pool_corpus_s3"].startswith("s3://bucket/smoke/evaluation/")
    assert result["base_top10_s3"].startswith("s3://bucket/smoke/evaluation/")
    assert result["ddb_table_name"] == "smoke-EvaluationResults"
    keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
    assert any(k.endswith("queries.jsonl") for k in keys)
    assert any(k.endswith("pool_corpus.jsonl") for k in keys)
    assert any(k.endswith("base_top10.jsonl") for k in keys)
