"""SageMaker entry point — Stage C: judge + NDCG + DDB report.

Reads queries / pool_corpus / base_top10 / ft_top10 from S3 (via --*-s3 args),
calls Claude Haiku per (query, doc) pair, computes graded NDCG@K and bootstrap
CI on the uplift, writes evaluation_report.json to S3 and a row to DynamoDB.

Designed to run in any container that has boto3 (SM CPU job today, ECS later).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse

import boto3

from bedrock_judge import judge_pairs
from eval_metrics import ndcg_at_k_graded, paired_bootstrap_ci

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"not an s3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _aws_region() -> str | None:
    """Resolve AWS region from env. SageMaker training containers set
    AWS_DEFAULT_REGION but not AWS_REGION; boto3 reads either."""
    import os
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def _build_s3_client():
    return boto3.client("s3", region_name=_aws_region())


def _build_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=_aws_region())


def _build_ddb_resource():
    return boto3.resource("dynamodb", region_name=_aws_region())


def _read_jsonl(s3, uri: str) -> list[dict]:
    bucket, key = parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    # NDJSON is strictly newline-delimited; do NOT use str.splitlines() because
    # it also splits on Unicode line separators (e.g. U+2028) that legitimately
    # appear inside JSON-escaped doc text and would break a single record into
    # multiple "lines".
    return [json.loads(l) for l in text.split("\n") if l.strip()]


def _put_json(s3, uri: str, payload: dict) -> None:
    bucket, key = parse_s3_uri(uri)
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C: judge + NDCG + DDB")
    p.add_argument("--queries-s3", required=True)
    p.add_argument("--pool-s3", required=True)
    p.add_argument("--base-top10-s3", required=True)
    p.add_argument("--ft-top10-s3", required=True)
    p.add_argument("--judge-model-id", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--judge-concurrency", type=int, default=16)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddb-table-name", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--out-s3", required=True)
    p.add_argument("--base-model-id", default="")
    return p.parse_args(argv)


def build_judge_pairs(
    queries: list[dict],
    base_top10: list[dict],
    ft_top10: list[dict],
    doc_text_by_id: dict[str, str],
) -> list[dict]:
    """Union of base & ft top-K per query → deduped (query, doc_id, doc_text)."""
    base_by_q = {b["query"]: b["top10"] for b in base_top10}
    ft_by_q = {f["query"]: f["top10"] for f in ft_top10}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for q in queries:
        for d_id in list(base_by_q.get(q["query"], [])) + list(ft_by_q.get(q["query"], [])):
            key = (q["query"], d_id)
            if key in seen:
                continue
            seen.add(key)
            text = doc_text_by_id.get(d_id, "")
            out.append({"query": q["query"], "doc_id": d_id, "doc_text": text})
    return out


def _to_dynamodb_safe(obj):
    """DynamoDB doesn't accept floats — coerce recursively to Decimal."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamodb_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamodb_safe(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    s3 = _build_s3_client()

    queries = _read_jsonl(s3, args.queries_s3)
    pool = _read_jsonl(s3, args.pool_s3)
    base_top10 = _read_jsonl(s3, args.base_top10_s3)
    ft_top10 = _read_jsonl(s3, args.ft_top10_s3)
    logger.info("Loaded queries=%d pool=%d base_top10=%d ft_top10=%d",
                len(queries), len(pool), len(base_top10), len(ft_top10))

    doc_text = {d["id"]: d["text"] for d in pool}
    pairs = build_judge_pairs(queries, base_top10, ft_top10, doc_text)
    logger.info("Judging %d pairs", len(pairs))

    bedrock = _build_bedrock_client()
    scores = judge_pairs(
        bedrock, args.judge_model_id, pairs,
        concurrency=args.judge_concurrency,
    )
    logger.info("Got %d judgments", len(scores))

    base_by_q = {b["query"]: b["top10"] for b in base_top10}
    ft_by_q = {f["query"]: f["top10"] for f in ft_top10}
    base_ndcg: list[float] = []
    ft_ndcg: list[float] = []
    for q in queries:
        per_doc = {d_id: scores.get((q["query"], d_id), 0.0)
                   for d_id in set(base_by_q.get(q["query"], []))
                              | set(ft_by_q.get(q["query"], []))}
        base_ndcg.append(ndcg_at_k_graded(base_by_q.get(q["query"], []), per_doc, args.top_k))
        ft_ndcg.append(ndcg_at_k_graded(ft_by_q.get(q["query"], []), per_doc, args.top_k))

    ci = paired_bootstrap_ci(
        base_ndcg, ft_ndcg,
        num_resamples=args.bootstrap_resamples, seed=args.seed,
    )

    metric_key = f"ndcg_at_{args.top_k}"
    report = {
        "task_id": args.task_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model_id": args.base_model_id,
        "num_eval_queries": len(queries),
        "candidate_pool_size_unique": len(pool),
        "judge_pairs_count": len(scores),
        "judge_model_id": args.judge_model_id,
        "metrics": {
            metric_key: {
                "base":          sum(base_ndcg) / max(len(base_ndcg), 1),
                "fine_tuned":    sum(ft_ndcg) / max(len(ft_ndcg), 1),
                "uplift_mean":   ci.mean_diff,
                "uplift_ci_low": ci.ci_low,
                "uplift_ci_high": ci.ci_high,
            }
        },
        "notes": [
            "Base top-K from AOSS kNN search. FT ranked within the same candidate pool.",
            "LLM judge (Claude Haiku) is the ground truth; eval queries were sampled "
            "from the synthetic training query pool.",
        ],
    }

    _put_json(s3, args.out_s3, report)
    logger.info("Wrote evaluation_report.json to %s", args.out_s3)

    table = _build_ddb_resource().Table(args.ddb_table_name)
    table.put_item(Item=_to_dynamodb_safe(report))
    logger.info("Wrote DDB row task_id=%s", args.task_id)


if __name__ == "__main__":
    main()
