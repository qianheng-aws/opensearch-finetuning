"""Prepare Evaluation Lambda — Stage A.

Reads eval_queries.jsonl from the ProcessBedrockOutput tarball, samples
queries stratified by source doc_id, runs kNN against the customer's AOSS
index for each query (top-N pool_size), aggregates a candidate pool and
per-query base top-K. Writes three JSONL files to S3 and returns their URIs.
"""

from __future__ import annotations

import io
import json
import logging
import random
import tarfile
from collections import defaultdict
from urllib.parse import urlparse

import boto3

from aoss_signer import signed_post

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def extract_eval_queries_from_tarball(tarbytes: bytes) -> list[dict]:
    """Open a model.tar.gz, return parsed records from eval_queries.jsonl."""
    with tarfile.open(fileobj=io.BytesIO(tarbytes), mode="r:gz") as tar:
        try:
            member = tar.getmember("eval_queries.jsonl")
        except KeyError:
            raise ValueError(
                "eval_queries.jsonl not found in tarball — "
                "process_output.py must run with --eval-fraction > 0"
            )
        f = tar.extractfile(member)
        if f is None:
            raise ValueError("eval_queries.jsonl is not a regular file")
        text = f.read().decode("utf-8")
    out: list[dict] = []
    # NDJSON is strictly newline-delimited; using str.splitlines() also splits
    # on Unicode line separators (U+2028 etc) and would corrupt records.
    for line_num, line in enumerate(text.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "query" not in rec or "doc_id" not in rec:
            raise ValueError(
                f"eval_queries line {line_num} requires 'query' and 'doc_id'")
        out.append({"query": rec["query"], "doc_id": rec["doc_id"]})
    return out


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"not an s3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _build_s3_client():
    return boto3.client("s3")


def sample_stratified_queries(pool: list[dict], num: int, seed: int) -> list[dict]:
    """Round-robin across doc_ids, deterministic with `seed`."""
    if num <= 0:
        return []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        by_doc[q["doc_id"]].append(q)

    doc_ids = sorted(by_doc.keys())
    rng = random.Random(seed + 1)
    rng.shuffle(doc_ids)
    rng2 = random.Random(seed)
    for d in by_doc:
        rng2.shuffle(by_doc[d])

    out: list[dict] = []
    cursors: dict[str, int] = defaultdict(int)
    while len(out) < num:
        progressed = False
        for d in doc_ids:
            if cursors[d] < len(by_doc[d]):
                out.append(by_doc[d][cursors[d]])
                cursors[d] += 1
                progressed = True
                if len(out) == num:
                    break
        if not progressed:
            break
    return out


def build_kNN_query_body(query_text: str, text_field: str, size: int) -> dict:
    """Build a `_search` body that the AOSS search pipeline will auto-rewrite
    into a kNN query. We use a plain `match` against the text field — the
    semantic_search_rewrite_processor on the search pipeline transforms it
    into a kNN query against the embedding field.
    """
    return {
        "size": size,
        "query": {"match": {text_field: query_text}},
    }


def parse_search_hits(response: dict, text_field: str) -> list[dict]:
    """Extract {id, text} from each hit; skip hits without the text field."""
    out: list[dict] = []
    for hit in response.get("hits", {}).get("hits", []):
        src = hit.get("_source") or {}
        text = src.get(text_field)
        if text is None:
            continue
        out.append({"id": hit["_id"], "text": text})
    return out


def _aoss_search(endpoint: str, index: str, body: dict) -> dict:
    """Production wrapper; tests monkey-patch this."""
    return signed_post(endpoint, f"/{index}/_search", body, service="aoss")


def handler(event: dict, context) -> dict:
    logger.info("PrepareEvaluation event: %s",
                json.dumps({k: v for k, v in event.items() if k != "context"}))

    task_id = event["task_id"]
    model_name = event["model_name"]
    aoss_endpoint = event["opensearch_endpoint"]
    aoss_index = event["opensearch_index_name"]
    eval_queries_s3 = event["eval_queries_s3"]
    bucket = event["data_bucket"]
    prefix = event["data_prefix"].rstrip("/")
    num_eval = int(event.get("num_eval_queries", 100))
    pool_size = int(event.get("pool_size", 100))
    top_k = int(event.get("top_k", 10))
    seed = int(event.get("seed", 42))
    # Accept either text_field (single) or text_fields (comma-separated string,
    # matching how upstream Lambdas thread the SFN $.text_fields parameter).
    # We use the first field for kNN search; the AOSS search pipeline rewrites
    # a single match query to kNN against the corresponding embedding field.
    text_fields_raw = event.get("text_fields") or event.get("text_field") or "text"
    text_field = text_fields_raw.split(",")[0].strip() or "text"

    s3 = _build_s3_client()

    src_bucket, src_key = parse_s3_uri(eval_queries_s3)
    obj = s3.get_object(Bucket=src_bucket, Key=src_key)
    tarbytes = obj["Body"].read()
    full_pool = extract_eval_queries_from_tarball(tarbytes)
    logger.info("Loaded %d candidate queries from tarball", len(full_pool))

    queries = sample_stratified_queries(full_pool, num_eval, seed)
    logger.info("Sampled %d eval queries", len(queries))

    pool_docs: dict[str, dict] = {}
    base_top10: list[dict] = []
    for q in queries:
        body = build_kNN_query_body(q["query"], text_field, pool_size)
        resp = _aoss_search(aoss_endpoint, aoss_index, body)
        hits = parse_search_hits(resp, text_field)
        for h in hits:
            pool_docs.setdefault(h["id"], h)
        base_top10.append({
            "query": q["query"],
            "top10": [h["id"] for h in hits[:top_k]],
        })

    logger.info("Candidate pool unique=%d, base_top10 entries=%d",
                len(pool_docs), len(base_top10))

    queries_key = f"{prefix}/queries.jsonl"
    pool_key = f"{prefix}/pool_corpus.jsonl"
    base_key = f"{prefix}/base_top10.jsonl"

    def _put(key: str, lines: list[str]) -> None:
        body = ("\n".join(lines) + "\n").encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/x-ndjson")

    _put(queries_key, [json.dumps(q, ensure_ascii=False) for q in queries])
    _put(pool_key, [json.dumps(d, ensure_ascii=False) for d in pool_docs.values()])
    _put(base_key, [json.dumps(b, ensure_ascii=False) for b in base_top10])

    return {
        "queries_s3": f"s3://{bucket}/{queries_key}",
        "pool_corpus_s3": f"s3://{bucket}/{pool_key}",
        "base_top10_s3": f"s3://{bucket}/{base_key}",
        "ddb_table_name": f"{model_name}-EvaluationResults",
        "candidate_pool_size_unique": len(pool_docs),
        "num_eval_queries": len(queries),
    }
