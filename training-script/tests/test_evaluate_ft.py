from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from evaluate_ft import (
    compute_top_k_rankings,
    parse_args,
    read_jsonl_from_s3,
    main,
)


def _arr(rows):
    return np.array(rows, dtype=np.float32)


def test_parse_args_required():
    args = parse_args([
        "--queries-s3", "s3://b/q",
        "--pool-s3", "s3://b/p",
        "--adapter-s3", "s3://b/a",
        "--base-model-id", "BAAI/bge-m3",
        "--out-s3", "s3://b/o",
    ])
    assert args.queries_s3 == "s3://b/q"
    assert args.top_k == 10
    assert args.max_seq_length == 512


def test_read_jsonl_from_s3():
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(
        (json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n").encode()
    )}
    out = read_jsonl_from_s3(s3, "s3://bucket/key")
    assert out == [{"a": 1}, {"a": 2}]
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_compute_top_k_orders_by_cosine():
    encoder = MagicMock()
    encoder.encode.side_effect = [
        _arr([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        _arr([[0.9, 0.1, 0]]),
    ]
    docs = [{"id": "d0", "text": "x"}, {"id": "d1", "text": "y"}, {"id": "d2", "text": "z"}]
    queries = [{"query": "q", "doc_id": "d0"}]
    out = compute_top_k_rankings(encoder, docs, queries, top_k=2)
    assert out == [["d0", "d1"]]


def test_compute_top_k_truncates():
    encoder = MagicMock()
    encoder.encode.side_effect = [_arr([[1, 0]] * 5), _arr([[1, 0]])]
    out = compute_top_k_rankings(
        encoder,
        [{"id": f"d{i}", "text": "t"} for i in range(5)],
        [{"query": "q", "doc_id": "d0"}],
        top_k=3,
    )
    assert len(out[0]) == 3


def test_main_writes_ft_top10_to_s3(monkeypatch, tmp_path):
    s3 = MagicMock()

    queries_body = (json.dumps({"query": "q1", "doc_id": "d_a"}) + "\n"
                    + json.dumps({"query": "q2", "doc_id": "d_c"}) + "\n").encode()
    pool_body = (json.dumps({"id": "d_a", "text": "a"}) + "\n"
                 + json.dumps({"id": "d_b", "text": "b"}) + "\n"
                 + json.dumps({"id": "d_c", "text": "c"}) + "\n").encode()

    def get_object(Bucket, Key):
        if Key.endswith("queries.jsonl"):
            return {"Body": io.BytesIO(queries_body)}
        if Key.endswith("pool.jsonl"):
            return {"Body": io.BytesIO(pool_body)}
        raise KeyError(Key)
    s3.get_object.side_effect = get_object
    s3.put_object = MagicMock()

    encoder = MagicMock()
    encoder.encode.side_effect = [
        _arr([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # docs
        _arr([[1, 0, 0], [0, 0, 1]]),             # queries
    ]

    monkeypatch.setattr("evaluate_ft._build_s3_client", lambda: s3)
    monkeypatch.setattr("evaluate_ft._download_and_extract_adapter",
                        lambda s3c, uri, tgt: None)
    monkeypatch.setattr("evaluate_ft.load_finetuned_encoder",
                        lambda base_model_id, adapter_path, max_seq_length: encoder)

    main([
        "--queries-s3", "s3://b/queries.jsonl",
        "--pool-s3", "s3://b/pool.jsonl",
        "--adapter-s3", "s3://b/model.tar.gz",
        "--base-model-id", "BAAI/bge-m3",
        "--top-k", "3",
        "--out-s3", "s3://b/ft_top10.jsonl",
    ])

    put_calls = [c for c in s3.put_object.call_args_list
                 if c.kwargs["Key"] == "ft_top10.jsonl"]
    assert len(put_calls) == 1
    body = put_calls[0].kwargs["Body"].decode()
    lines = [json.loads(l) for l in body.strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["query"] == "q1"
    assert lines[0]["top10"][0] == "d_a"
    assert lines[1]["query"] == "q2"
    assert lines[1]["top10"][0] == "d_c"
