"""Local Stage A + B: produce queries.jsonl, pool_corpus.jsonl,
base_top10.jsonl, ft_top10.jsonl from yuanchu data + LoRA adapter.

Stage A (normally a Lambda doing AOSS kNN): we replace AOSS with a local
SentenceTransformer load of the BAAI/bge-m3 base model, encode the yuanchu
corpus + queries, and produce base_top10 by cosine.

Stage B (normally an SM GPU job): merge LoRA into base, encode again, produce
ft_top10.

We then upload the 4 JSONL files to S3 so Stage C (cloud SM job) can read them.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

DATA_DIR = Path("/tmp/eval-poc-c")
ADAPTER_DIR = DATA_DIR / "adapter"
OUT_DIR = DATA_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

BASE_MODEL_ID = "Alibaba-NLP/gte-multilingual-base"
TOP_K = 10  # all 10 docs, since corpus is exactly 10
BATCH_SIZE = 8


def load_corpus():
    docs = []
    with open(DATA_DIR / "corpus.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append({"id": rec["id"], "text": rec["text"]})
    return docs


def load_queries():
    """yuanchu queries.jsonl is { doc_id, text, queries: [str, ...] }.
    Flatten into [{query, doc_id}, ...]."""
    out = []
    with open(DATA_DIR / "queries.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for q in rec["queries"]:
                out.append({"query": q, "doc_id": rec["doc_id"]})
    return out


def encode_with_st_model(model, texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
    ).astype(np.float32)


def compute_top_k(encoder, docs, queries, k):
    print(f"  encoding {len(docs)} docs ...")
    doc_vecs = encode_with_st_model(encoder, [d["text"] for d in docs])
    print(f"  encoding {len(queries)} queries ...")
    q_vecs = encode_with_st_model(encoder, [q["query"] for q in queries])
    print(f"  doc_vecs shape: {doc_vecs.shape}, q_vecs shape: {q_vecs.shape}")
    sims = q_vecs @ doc_vecs.T
    top_idx = np.argsort(-sims, axis=1)[:, :k]
    doc_ids = [d["id"] for d in docs]
    rankings = [[doc_ids[i] for i in row] for row in top_idx]
    return rankings


def load_finetuned_encoder(adapter_path: Path) -> SentenceTransformer:
    """Replicate evaluate_ft.load_finetuned_encoder logic."""
    print(f"  loading base AutoModel: {BASE_MODEL_ID}")
    base = AutoModel.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    print(f"  applying LoRA adapter from {adapter_path}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    print("  merging LoRA into base ...")
    merged = peft_model.merge_and_unload()

    tmp = tempfile.mkdtemp(prefix="ft-merged-")
    print(f"  saving merged to {tmp}")
    merged.save_pretrained(tmp)
    for sub in (
        "modules.json", "1_Pooling", "2_Normalize",
        "config_sentence_transformers.json", "sentence_bert_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ):
        src = adapter_path / sub
        dst = Path(tmp) / sub
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)

    print(f"  loading SentenceTransformer from {tmp}")
    model = SentenceTransformer(tmp, trust_remote_code=True)
    return model


def main():
    docs = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(docs)} docs and {len(queries)} queries")
    print(f"Sample query: {queries[0]}")
    print()

    # === Stage A: base ===
    print("=== Stage A: base model (BAAI/bge-m3) ===")
    base_encoder = SentenceTransformer(BASE_MODEL_ID, trust_remote_code=True)
    base_rankings = compute_top_k(base_encoder, docs, queries, TOP_K)
    del base_encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()

    # === Stage B: ft (base + LoRA) ===
    print("=== Stage B: fine-tuned (base + LoRA) ===")
    ft_encoder = load_finetuned_encoder(ADAPTER_DIR)
    ft_rankings = compute_top_k(ft_encoder, docs, queries, TOP_K)
    del ft_encoder
    print()

    # === Write 4 JSONL files ===
    print("=== Writing 4 JSONL files ===")
    with open(OUT_DIR / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with open(OUT_DIR / "pool_corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(OUT_DIR / "base_top10.jsonl", "w") as f:
        for q, r in zip(queries, base_rankings):
            f.write(json.dumps({"query": q["query"], "top10": r}) + "\n")
    with open(OUT_DIR / "ft_top10.jsonl", "w") as f:
        for q, r in zip(queries, ft_rankings):
            f.write(json.dumps({"query": q["query"], "top10": r}) + "\n")

    print(f"  queries.jsonl    : {(OUT_DIR / 'queries.jsonl').stat().st_size}B")
    print(f"  pool_corpus.jsonl: {(OUT_DIR / 'pool_corpus.jsonl').stat().st_size}B")
    print(f"  base_top10.jsonl : {(OUT_DIR / 'base_top10.jsonl').stat().st_size}B")
    print(f"  ft_top10.jsonl   : {(OUT_DIR / 'ft_top10.jsonl').stat().st_size}B")
    print()

    # Sanity: how do the rankings differ?
    print("=== Ranking diff diagnostics ===")
    for i, (q, br, fr) in enumerate(zip(queries, base_rankings, ft_rankings)):
        match = sum(1 for a, b in zip(br, fr) if a == b)
        gold_in_base_top1 = br[0] == q["doc_id"]
        gold_in_ft_top1 = fr[0] == q["doc_id"]
        print(f"  Q{i+1} '{q['query'][:50]}' (gold={q['doc_id']})")
        print(f"     base top1={br[0]}, ft top1={fr[0]}, base==ft positions: {match}/10")


if __name__ == "__main__":
    main()
