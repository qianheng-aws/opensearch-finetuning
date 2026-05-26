from __future__ import annotations

import pytest

from process_output import stratified_split


def test_stratified_split_zero_eval_fraction_keeps_all_in_train():
    pairs = [
        {"query": "q1", "doc_id": "d1", "doc_text": "t1"},
        {"query": "q2", "doc_id": "d1", "doc_text": "t1"},
        {"query": "q3", "doc_id": "d2", "doc_text": "t2"},
    ]
    train, evalp = stratified_split(pairs, eval_fraction=0.0, seed=42)
    assert len(train) == 3
    assert evalp == []


def test_stratified_split_full_eval_fraction_keeps_all_in_eval():
    pairs = [
        {"query": "q1", "doc_id": "d1", "doc_text": "t1"},
        {"query": "q2", "doc_id": "d2", "doc_text": "t2"},
    ]
    train, evalp = stratified_split(pairs, eval_fraction=1.0, seed=42)
    assert train == []
    assert len(evalp) == 2


def test_stratified_split_20_percent_per_doc():
    pairs = []
    for d in ("d1", "d2"):
        for i in range(5):
            pairs.append({"query": f"{d}_q{i}", "doc_id": d, "doc_text": f"t_{d}"})

    train, evalp = stratified_split(pairs, eval_fraction=0.2, seed=42)

    train_d1 = [p for p in train if p["doc_id"] == "d1"]
    train_d2 = [p for p in train if p["doc_id"] == "d2"]
    eval_d1 = [p for p in evalp if p["doc_id"] == "d1"]
    eval_d2 = [p for p in evalp if p["doc_id"] == "d2"]

    assert len(train_d1) == 4 and len(eval_d1) == 1
    assert len(train_d2) == 4 and len(eval_d2) == 1


def test_stratified_split_is_deterministic_with_same_seed():
    pairs = []
    for d in ("d1", "d2", "d3"):
        for i in range(4):
            pairs.append({"query": f"{d}_q{i}", "doc_id": d, "doc_text": "t"})

    t1, e1 = stratified_split(pairs, eval_fraction=0.25, seed=99)
    t2, e2 = stratified_split(pairs, eval_fraction=0.25, seed=99)
    assert [p["query"] for p in t1] == [p["query"] for p in t2]
    assert [p["query"] for p in e1] == [p["query"] for p in e2]


def test_stratified_split_doc_with_one_query_falls_to_train_at_low_fraction():
    pairs = [{"query": "q1", "doc_id": "d_single", "doc_text": "t"}]
    train, evalp = stratified_split(pairs, eval_fraction=0.2, seed=1)
    assert len(train) == 1
    assert evalp == []


def test_stratified_split_partition_is_complete():
    pairs = []
    for d in ("d1", "d2", "d3"):
        for i in range(7):
            pairs.append({"query": f"{d}_q{i}", "doc_id": d, "doc_text": "t"})
    train, evalp = stratified_split(pairs, eval_fraction=0.3, seed=11)
    assert len(train) + len(evalp) == len(pairs)
    train_q = {p["query"] for p in train}
    eval_q = {p["query"] for p in evalp}
    assert train_q.isdisjoint(eval_q)
    assert train_q | eval_q == {p["query"] for p in pairs}
