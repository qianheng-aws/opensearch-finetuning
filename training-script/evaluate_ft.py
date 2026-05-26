"""SageMaker entry point — Stage B: embed pool + queries with LoRA-merged
fine-tuned model, write per-query top-K ranking to S3.

Inputs / outputs are passed via S3 URIs as command-line args; this avoids
SM-channel auto-extraction and keeps the script portable to ECS.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"not an s3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _build_s3_client():
    import os
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return boto3.client("s3", region_name=region)


def read_jsonl_from_s3(s3, uri: str) -> list[dict]:
    bucket, key = parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    # NDJSON is strictly newline-delimited; do NOT use str.splitlines() because
    # it also splits on Unicode line separators (e.g. U+2028) that legitimately
    # appear inside JSON-escaped doc text and would break a single record into
    # multiple "lines".
    return [json.loads(l) for l in text.split("\n") if l.strip()]


def write_jsonl_to_s3(s3, uri: str, records: list[dict]) -> None:
    bucket, key = parse_s3_uri(uri)
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/x-ndjson")


def _download_and_extract_adapter(s3, adapter_s3: str, target_dir: str) -> None:
    """Download train job's model.tar.gz from S3 and extract into target_dir."""
    bucket, key = parse_s3_uri(adapter_s3)
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(target_dir, filter="data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B: FT embed + rank")
    p.add_argument("--queries-s3", required=True)
    p.add_argument("--pool-s3", required=True)
    p.add_argument("--adapter-s3", required=True)
    p.add_argument("--base-model-id", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--out-s3", required=True)
    return p.parse_args(argv)


class _Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def compute_top_k_rankings(
    encoder: _Encoder,
    docs: list[dict],
    queries: list[dict],
    top_k: int,
) -> list[list[str]]:
    doc_vecs = encoder.encode([d["text"] for d in docs])
    query_vecs = encoder.encode([q["query"] for q in queries])
    sims = query_vecs @ doc_vecs.T
    top_idx = np.argsort(-sims, axis=1)[:, :top_k]
    doc_ids = [d["id"] for d in docs]
    return [[doc_ids[i] for i in row] for row in top_idx]


def load_finetuned_encoder(base_model_id: str, adapter_path: str, max_seq_length: int):
    """Production loader; tests monkey-patch this. Lazy imports so tests don't
    require peft/sentence-transformers/transformers to be installed."""
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel

    base = AutoModel.from_pretrained(base_model_id, trust_remote_code=False)
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()

    # NOTE: We deliberately use mkdtemp (no auto-cleanup) instead of
    # TemporaryDirectory because SentenceTransformer may lazy-load tokenizer
    # files after construction; deleting the dir while the encoder is alive
    # would break later .encode() calls. The container reclaims this on exit.
    tmp = tempfile.mkdtemp(prefix="ft-merged-")
    merged.save_pretrained(tmp)
    for sub in (
        "modules.json", "1_Pooling", "2_Normalize",
        "config_sentence_transformers.json", "sentence_bert_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ):
        src = Path(adapter_path) / sub
        dst = Path(tmp) / sub
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)
    model = SentenceTransformer(tmp, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return _SentenceTransformerEncoder(model)


class _SentenceTransformerEncoder:
    def __init__(self, model):
        self._model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False, batch_size=32,
        ).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    s3 = _build_s3_client()

    queries = read_jsonl_from_s3(s3, args.queries_s3)
    pool = read_jsonl_from_s3(s3, args.pool_s3)
    logger.info("Loaded %d queries, %d pool docs", len(queries), len(pool))

    with tempfile.TemporaryDirectory() as adapter_dir:
        _download_and_extract_adapter(s3, args.adapter_s3, adapter_dir)
        encoder = load_finetuned_encoder(args.base_model_id, adapter_dir, args.max_seq_length)
        rankings = compute_top_k_rankings(encoder, pool, queries, args.top_k)

    out = [{"query": q["query"], "top10": r} for q, r in zip(queries, rankings)]
    write_jsonl_to_s3(s3, args.out_s3, out)
    logger.info("Wrote %d FT rankings to %s", len(out), args.out_s3)


if __name__ == "__main__":
    main()
