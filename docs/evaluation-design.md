# Fine-Tune Evaluation Design (POC v3 — kNN-prep + FT-embed + judge)

**Goal:** Add a 3-stage `Evaluation` pipeline to the OpenSearch fine-tuning POC that compares the LoRA-fine-tuned model to the customer's existing base dense retriever on their own data, judged by Claude Haiku, with the resulting NDCG@10 written to DynamoDB.

**Architecture:**

```
Stage A: PrepareEvaluation  (Lambda)
  - read eval_queries.jsonl from S3, sample 100 stratified by doc_id
  - for each query: kNN top-100 against customer's AOSS index
    (search pipeline auto-embeds query via base model connector)
  - write queries.jsonl, pool_corpus.jsonl, base_top10.jsonl to S3

Stage B: EvaluationFTEmbed  (SageMaker GPU job, ml.g5.2xlarge)
  - load LoRA-merged FT model (peft + base SentenceTransformer)
  - embed pool corpus + queries; rank within pool → ft_top10
  - write ft_top10.jsonl to S3

Stage C: EvaluationJudgeReport  (SageMaker CPU job, ml.m5.xlarge)
  - read base_top10 + ft_top10 + pool_corpus
  - dedup judge pairs; call Claude Haiku judge per pair (16 parallel)
  - graded NDCG@10 per query × per config; paired bootstrap CI on uplift
  - write evaluation_report.json to S3 + put item to EvaluationResults DDB
```

All 3 stages communicate **only through explicit S3 URIs passed as command-line args / hyperparameters** — no SageMaker channel auto-extraction. This keeps Stage C portable: when prod migrates Stage C to ECS, the same `evaluate_judge.py` runs as `python -m evaluate_judge --base-top10-s3 ... --ft-top10-s3 ...` with zero code changes.

**Tech Stack:**
- Python 3.12 (Lambda + SM containers)
- AOSS via SigV4-signed `urllib.request` (matches existing POC Lambdas)
- sentence-transformers 5.0.0, peft 0.14.0, faiss-cpu (Stage B GPU container)
- boto3 — Bedrock InvokeModel + DDB PutItem + S3 (Stages A and C)
- pytest for unit tests
- CloudFormation / Step Functions

**Pre-conditions (verified outside this plan):**

1. The user has an AOSS collection with:
   - An index whose docs already have a base dense embedding field.
   - A search pipeline configured with `semantic_search_rewrite_processor` (or equivalent) so that a plain `_search` request with `match`-style or `neural` query is auto-rewritten by OpenSearch into a kNN search using the base model. (POC test endpoint `https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com` with index `test-ecs-v11` is the reference.)
2. The fine-tune workflow's `s3_corpus_path` is **the same set of docs** that exist in the AOSS index (i.e. the corpus the AOSS index was built from). Otherwise eval queries' `doc_id` won't match anything in AOSS. Plan v3 documents this contract; checking it is the user's responsibility at deploy time.
3. v1 Tasks 1–2 (`stratified_split` in `process_output.py` + `eval-fraction`/`seed` hyperparameters) are merged. If not, do them first — same instructions as v1 plan, ~30 min.
4. Bedrock Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) is enabled in the deploy region.

---

## Why this design

### Why three stages instead of one big SM job

| Stage | Workload | Dominated by | Right resource |
|---|---|---|---|
| A: Prepare | 100 kNN searches against AOSS, ~1s each | I/O (network) | Lambda |
| B: FT embed + rank | 10K-doc embed via 560M-param model | GPU compute | SM ml.g5.2xlarge (~5 min) |
| C: Judge + report | ~1500 Bedrock calls, 16 parallel | I/O (Bedrock) | SM ml.m5.xlarge (~12 min) |

A single SM GPU job covering all three would idle the GPU during ~12 min of Bedrock waiting (~$0.30 wasted but mainly: bad design separation). Splitting also means a Stage C failure (e.g. Bedrock throttling) doesn't waste the embedding work — SFN can rerun just Stage C from existing S3 artifacts.

### Why Stage C runs as SM CPU job (not Lambda or ECS) in POC

- **Lambda 15-min timeout is too tight.** 1500 pairs × p99 1s ÷ 16 parallel ≈ 94 s typical, but Bedrock throttling retries can push past the limit.
- **ECS would require new infra** (ECR repo, Docker image, ECS cluster, task def). POC has none of that. Adding it for one job is overkill.
- **SM CPU job (`ml.m5.xlarge`)** reuses POC's existing `training-script.tar.gz` packaging, the same `SageMakerTrainingRole`, and the same `sagemaker:createTrainingJob.sync` SFN integration. Zero new infrastructure surface.

When prod migrates this codebase to AOS repo (which already has ECS for fine-tune), Stage C's `evaluate_judge.py` runs as-is in an ECS container — see "Portability checklist" at the end.

### Why pool from base only (not base ∪ ft)

If FT would surface a doc that base wouldn't, that doc never enters the pool, so FT's NDCG is bounded above by base's pool. This is the standard pool-eval limitation (HLD §5.2). Adding "FT-side top-100 retrieval" too would close that gap but doubles the FT GPU work and requires FT to embed the entire corpus (not just the pool subset). Phase 2 work; out of scope here.

---

## File Structure

All paths relative to `/workplace/qianheng/FineTunePOC/opensearch-finetuning/`.

### Create

- `lambdas/prepare-evaluation/index.py` — kNN-driven pool/queries builder (Stage A)
- `lambdas/prepare-evaluation/aoss_signer.py` — SigV4-signed AOSS request helper (refactor of pattern in `lambdas/data-extractor/index.py`)
- `training-script/eval_metrics.py` — graded NDCG@K + paired bootstrap CI (pure)
- `training-script/bedrock_judge.py` — Claude Haiku judge wrapper (pure logic + `boto3` injectable)
- `training-script/evaluate_ft.py` — Stage B SM entry point (FT embed + rank)
- `training-script/evaluate_judge.py` — Stage C SM entry point (judge + NDCG + DDB)
- `training-script/tests/__init__.py`
- `training-script/tests/test_eval_metrics.py`
- `training-script/tests/test_bedrock_judge.py`
- `training-script/tests/test_evaluate_ft.py`
- `training-script/tests/test_evaluate_judge.py`
- `lambdas/prepare-evaluation/test_index.py`

### Modify

- `training-script/requirements.txt` — add `peft==0.14.0`
- `training-script/process_output.py` — `--eval-fraction` + `--seed` hyperparameters and `stratified_split` helper (carry-over from v1 Task 2; skip if already merged)
- `build.sh` — add `prepare-evaluation-lambda.zip` to the upload list
- `opensearch-finetune-poc.yaml`:
  - `EvaluationResultsTable` (DynamoDB)
  - `PrepareEvaluationLambdaZipUrl` parameter
  - `DownloadLambdaZips` entry for the new zip
  - `PrepareEvaluationLambda` resource
  - `SageMakerTrainingRole` — add `bedrock:InvokeModel` for Claude Haiku + `dynamodb:PutItem` on the new table
  - `LambdaExecutionRole` (or whichever role `prepare-evaluation` uses) — add `aoss:APIAccessAll` on the customer collection + `s3:PutObject` on the data bucket
  - 3 new SFN states: `PrepareEvaluation` → `StartSageMakerEvaluationFTEmbed` → `StartSageMakerEvaluationJudge`, inserted between `StartSageMakerTraining` and `CheckDeployEndpoint`
  - `eval-fraction: "0.2"` and `seed: "42"` on `ProcessBedrockOutput` (carry-over from v1 if not already there)

---

## Architecture Reference

### S3 URI conventions

All artifacts under `s3://<DataBucket>/<model_name>/evaluation/`:

| File | Producer | Consumers | Format |
|---|---|---|---|
| `queries.jsonl` | Stage A | B, C | `{query, doc_id}` per line — the 100 sampled queries |
| `pool_corpus.jsonl` | Stage A | B, C | `{id, text}` per line — union of all queries' top-100 (~10K unique) |
| `base_top10.jsonl` | Stage A | C | `{query, top10: [doc_id, ...]}` per line |
| `ft_top10.jsonl` | Stage B | C | same shape as base_top10 |
| `evaluation_report.json` | Stage C | — | the report (also written to DDB) |

### `evaluation_report.json` / DDB row schema

```json
{
  "task_id": "ft-2026-05-26-abcdef",
  "schema_version": 1,
  "generated_at": "2026-05-26T12:34:56Z",
  "base_model_id": "BAAI/bge-m3",
  "num_eval_queries": 100,
  "pool_size_per_query": 100,
  "candidate_pool_size_unique": 9842,
  "judge_pairs_count": 1573,
  "judge_model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
  "metrics": {
    "ndcg_at_10": {
      "base":          0.612,
      "fine_tuned":    0.687,
      "uplift_mean":   0.075,
      "uplift_ci_low": 0.041,
      "uplift_ci_high":0.108
    }
  },
  "notes": [
    "Base top-K from AOSS kNN search. FT ranked within the same candidate pool.",
    "LLM judge (Claude Haiku) is the ground truth; eval queries were sampled from the synthetic training query pool."
  ]
}
```

### Stage A inputs (Lambda event)

```json
{
  "task_id": "ft-2026-05-26-abcdef",
  "model_name": "eval-smoke",
  "opensearch_endpoint": "https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com",
  "opensearch_index_name": "test-ecs-v11",
  "eval_queries_s3": "s3://<bucket>/<model_name>/process-output/<exec>/output/model.tar.gz",
  "data_bucket": "<bucket>",
  "data_prefix": "<model_name>/evaluation",
  "num_eval_queries": 100,
  "pool_size": 100,
  "top_k": 10,
  "seed": 42,
  "text_field": "text"
}
```

### Stage B / C SM hyperparameters

Both are `sagemaker:createTrainingJob.sync` invocations with `sagemaker_program=evaluate_ft.py` / `evaluate_judge.py`. **Crucially, neither stage uses InputDataConfig channels.** All inputs come via hyperparameters as S3 URIs — that's the portability discipline.

Stage B (`evaluate_ft.py`):
```
--queries-s3        s3://.../queries.jsonl
--pool-s3           s3://.../pool_corpus.jsonl
--adapter-s3        s3://.../model-artifacts/finetune-<exec>/output/model.tar.gz
--base-model-id     BAAI/bge-m3
--top-k             10
--max-seq-length    512
--out-s3            s3://.../ft_top10.jsonl
```

Stage C (`evaluate_judge.py`):
```
--queries-s3         s3://.../queries.jsonl
--pool-s3            s3://.../pool_corpus.jsonl
--base-top10-s3      s3://.../base_top10.jsonl
--ft-top10-s3        s3://.../ft_top10.jsonl
--judge-model-id     us.anthropic.claude-haiku-4-5-20251001-v1:0
--judge-concurrency  16
--top-k              10
--bootstrap-resamples 1000
--seed               42
--ddb-table-name     <ModelName>-EvaluationResults
--task-id            $$.Execution.Name
--out-s3             s3://.../evaluation_report.json
```

---

## Task 1: `eval_metrics.py` — graded NDCG + paired bootstrap

**Files:**
- Create: `training-script/eval_metrics.py`
- Create: `training-script/tests/__init__.py` (empty)
- Create: `training-script/tests/test_eval_metrics.py`

Pure functions. The judge produces graded scores in `[0, 1]`; NDCG must use them directly as gains:
- `DCG@K  = Σ rel(d_i) / log2(i + 2)` for `i ∈ [0, K)`
- `IDCG@K = DCG@K` of the same K candidates sorted by rel desc
- `NDCG@K = DCG@K / IDCG@K` (0 if `IDCG@K == 0`)

- [ ] **Step 1: Write failing tests**

Create `training-script/tests/__init__.py` (empty). Create `training-script/tests/test_eval_metrics.py`:

```python
from __future__ import annotations

import math

import pytest

from eval_metrics import ndcg_at_k_graded, paired_bootstrap_ci


def test_ndcg_perfect_ranking_is_one():
    ranking = ["d_a", "d_b", "d_c"]
    scores = {"d_a": 1.0, "d_b": 0.8, "d_c": 0.5}
    assert ndcg_at_k_graded(ranking, scores, k=3) == pytest.approx(1.0)


def test_ndcg_zero_relevant_returns_zero():
    assert ndcg_at_k_graded(["d_a"], {"d_a": 0.0}, k=1) == 0.0


def test_ndcg_imperfect_ranking_below_one():
    ranking = ["d_b", "d_a", "d_c"]
    scores = {"d_a": 1.0, "d_b": 0.5, "d_c": 0.0}
    val = ndcg_at_k_graded(ranking, scores, k=3)
    assert 0.0 < val < 1.0


def test_ndcg_missing_doc_treated_as_zero():
    ranking = ["d_x", "d_a"]
    scores = {"d_a": 1.0}
    expected_dcg = 0.0 + 1.0 / math.log2(3)
    expected_idcg = 1.0
    assert ndcg_at_k_graded(ranking, scores, k=2) == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_truncates_to_k_for_both_dcg_and_idcg():
    ranking = ["d_a", "d_b", "d_c", "d_d"]
    scores = {"d_a": 0.3, "d_b": 0.6, "d_c": 0.9, "d_d": 1.0}
    val = ndcg_at_k_graded(ranking, scores, k=2)
    expected_dcg = 0.3 + 0.6 / math.log2(3)
    expected_idcg = 1.0 + 0.9 / math.log2(3)
    assert val == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_empty_ranking_returns_zero():
    assert ndcg_at_k_graded([], {"d_a": 1.0}, k=10) == 0.0


def test_paired_bootstrap_constant_uplift():
    base = [0.4, 0.5, 0.6, 0.7, 0.8] * 20
    ft = [b + 0.1 for b in base]
    r = paired_bootstrap_ci(base, ft, num_resamples=500, seed=42)
    assert r.mean_diff == pytest.approx(0.1, abs=1e-9)
    assert r.ci_low == pytest.approx(0.1, abs=1e-9)
    assert r.ci_high == pytest.approx(0.1, abs=1e-9)


def test_paired_bootstrap_zero_diff_includes_zero():
    r = paired_bootstrap_ci([0.5] * 50, [0.5] * 50, num_resamples=500, seed=42)
    assert r.mean_diff == 0.0
    assert r.ci_low <= 0.0 <= r.ci_high


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([0.1, 0.2], [0.1], num_resamples=10, seed=1)


def test_paired_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_ci([], [], num_resamples=10, seed=1)
```

- [ ] **Step 2: Run tests and verify they fail**

```
cd training-script
PYTHONPATH=. pytest tests/test_eval_metrics.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `eval_metrics.py`**

Create `training-script/eval_metrics.py`:

```python
"""Graded NDCG@K and paired-bootstrap CI for the evaluation report.

Both functions are pure (no I/O, no torch). Suitable to import from
the SM judge job (CPU) and from any future ECS migration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k_graded(
    ranking: list[str],
    scores: dict[str, float],
    k: int,
) -> float:
    if not ranking:
        return 0.0
    ranked = [scores.get(d, 0.0) for d in ranking[:k]]
    dcg = _dcg(ranked, k)
    ideal = sorted(ranked, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float


def paired_bootstrap_ci(
    base_per_query: list[float],
    ft_per_query: list[float],
    num_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapResult:
    if len(base_per_query) != len(ft_per_query):
        raise ValueError("base and ft must be the same length")
    n = len(base_per_query)
    if n == 0:
        raise ValueError("inputs must not be empty")

    diffs = [ft - b for b, ft in zip(base_per_query, ft_per_query)]
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(num_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        sample_means.append(s / n)

    sample_means.sort()
    lo_idx = int((alpha / 2) * num_resamples)
    hi_idx = int((1 - alpha / 2) * num_resamples) - 1
    return BootstrapResult(
        mean_diff=mean_diff,
        ci_low=sample_means[max(0, lo_idx)],
        ci_high=sample_means[min(num_resamples - 1, hi_idx)],
    )
```

- [ ] **Step 4: Run tests, verify pass**

```
PYTHONPATH=. pytest tests/test_eval_metrics.py -v
```
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
git add training-script/eval_metrics.py training-script/tests/__init__.py training-script/tests/test_eval_metrics.py
git commit -m "feat(training): graded NDCG + paired bootstrap CI"
```

---

## Task 2: `bedrock_judge.py` — Claude Haiku judge wrapper

**Files:**
- Create: `training-script/bedrock_judge.py`
- Create: `training-script/tests/test_bedrock_judge.py`

The judge module: one entry point `judge_pairs(client, model_id, pairs, concurrency, max_retries=5)`. Each pair is `{query, doc_id, doc_text}`; returns `{(query, doc_id): score}`. Tests inject a `MagicMock` Bedrock client.

Key design points:
- Output parser is strict — a single float in `[0, 1]`, clamped.
- Retries on `ClientError` with exponential backoff.
- `ThreadPoolExecutor` for concurrency (each pair is one Bedrock call).
- Dedupes input pairs.

- [ ] **Step 1: Write failing tests**

Create `training-script/tests/test_bedrock_judge.py`:

```python
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


def test_judge_pairs_retries_on_throttling():
    from botocore.exceptions import ClientError
    client = MagicMock()
    err = ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
    client.invoke_model.side_effect = [err, _resp("0.7")]
    out = judge_pairs(client, "m", [
        {"query": "q", "doc_id": "d", "doc_text": "t"},
    ], concurrency=1, max_retries=3)
    assert out == {("q", "d"): 0.7}
    assert client.invoke_model.call_count == 2


def test_judge_pairs_gives_up_after_max_retries():
    from botocore.exceptions import ClientError
    client = MagicMock()
    err = ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
    client.invoke_model.side_effect = err
    with pytest.raises(JudgeError, match="exhausted"):
        judge_pairs(client, "m", [
            {"query": "q", "doc_id": "d", "doc_text": "t"},
        ], concurrency=1, max_retries=2)
```

- [ ] **Step 2: Verify tests fail**

```
PYTHONPATH=. pytest tests/test_bedrock_judge.py -v
```
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `bedrock_judge.py`**

Create `training-script/bedrock_judge.py`:

```python
"""Claude Haiku judge wrapper. One Bedrock call per (query, doc) pair."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a search relevance judge.

Rate how relevant the document is to the query, on a scale from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Query: {query}

Document: {doc_text}

Respond with a single number between 0.0 and 1.0. No explanation."""

_FLOAT_RE = re.compile(r"-?\d+(\.\d+)?")


class JudgeError(RuntimeError):
    pass


def build_prompt(query: str, doc_text: str) -> str:
    return PROMPT_TEMPLATE.format(query=query, doc_text=doc_text)


def parse_score(text: str) -> float:
    m = _FLOAT_RE.search(text)
    if not m:
        raise JudgeError(f"no numeric score: {text!r}")
    return max(0.0, min(1.0, float(m.group(0))))


def _invoke_with_retry(client, model_id: str, prompt: str, max_retries: int) -> float:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": prompt}],
    })
    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read().decode())
            return parse_score(payload["content"][0]["text"])
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 1.0 * (2 ** attempt)
                logger.warning("judge attempt %d/%d failed: %s, retrying in %.1fs",
                               attempt + 1, max_retries, e, wait)
                time.sleep(wait)
                continue
            raise JudgeError(f"retries exhausted: {e}") from e
    raise JudgeError("unreachable")


def judge_pairs(
    client,
    model_id: str,
    pairs: Iterable[dict],
    concurrency: int,
    max_retries: int = 5,
) -> dict[tuple[str, str], float]:
    seen: dict[tuple[str, str], dict] = {}
    for p in pairs:
        seen.setdefault((p["query"], p["doc_id"]), p)

    results: dict[tuple[str, str], float] = {}
    if not seen:
        return results

    def _one(kp: tuple[tuple[str, str], dict]) -> tuple[tuple[str, str], float]:
        key, p = kp
        return key, _invoke_with_retry(client, model_id, build_prompt(p["query"], p["doc_text"]), max_retries)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, kp) for kp in seen.items()]
        for fut in as_completed(futures):
            key, score = fut.result()
            results[key] = score
    return results
```

- [ ] **Step 4: Verify tests pass**

```
PYTHONPATH=. pytest tests/test_bedrock_judge.py -v
```
Expected: 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add training-script/bedrock_judge.py training-script/tests/test_bedrock_judge.py
git commit -m "feat(training): Bedrock InvokeModel judge wrapper"
```

---

## Task 3: `prepare-evaluation` Lambda — kNN against AOSS

**Files:**
- Create: `lambdas/prepare-evaluation/index.py`
- Create: `lambdas/prepare-evaluation/aoss_signer.py`
- Create: `lambdas/prepare-evaluation/test_index.py`

The Lambda is the only place that talks to the customer's AOSS collection. For each sampled query it does a `_search` request — the AOSS search pipeline (configured externally with `semantic_search_rewrite_processor`) auto-embeds the query using the customer's base model connector, returns top-100 hits with `_id` and `_source.<text_field>`.

Output to S3:
- `queries.jsonl` — `{query, doc_id}` for each of the 100 sampled queries
- `pool_corpus.jsonl` — `{id, text}` per unique doc across all top-100 hits (~10K docs)
- `base_top10.jsonl` — `{query, top10: [doc_id, ...]}` per query

Lambda return value (consumed by SFN):
```json
{
  "queries_s3": "s3://...",
  "pool_corpus_s3": "s3://...",
  "base_top10_s3": "s3://...",
  "ddb_table_name": "<ModelName>-EvaluationResults",
  "candidate_pool_size_unique": 9842,
  "num_eval_queries": 100
}
```

The eval_queries source is `eval_queries.jsonl` packaged inside `model.tar.gz` from `ProcessBedrockOutput`. Lambda downloads the tarball, extracts only that one file. (Lambda ephemeral storage is 512 MB by default, enough.)

- [ ] **Step 1: Write failing tests**

Create `lambdas/prepare-evaluation/test_index.py`:

```python
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
    # the query passes the raw query text; AOSS search pipeline rewrites it
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
    # Stub eval_queries tarball
    payload = "\n".join(
        json.dumps({"query": f"q{i}", "doc_id": f"d{i % 3}"}) for i in range(15)
    ).encode()
    tarbytes = _make_tarball_bytes({"eval_queries.jsonl": payload})

    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(tarbytes)}
    s3.put_object = MagicMock()

    aoss_search = MagicMock()
    # First query returns 2 hits; we don't care about exact contents — just that
    # handler aggregates them.
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
    # Wrote 3 distinct files
    keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
    assert any(k.endswith("queries.jsonl") for k in keys)
    assert any(k.endswith("pool_corpus.jsonl") for k in keys)
    assert any(k.endswith("base_top10.jsonl") for k in keys)
```

- [ ] **Step 2: Verify tests fail**

```
cd lambdas/prepare-evaluation
PYTHONPATH=. pytest test_index.py -v
```
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement `aoss_signer.py`**

Create `lambdas/prepare-evaluation/aoss_signer.py`:

```python
"""SigV4-signed AOSS request helper, mirroring the pattern used in
lambdas/data-extractor/index.py.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def signed_post(endpoint: str, path: str, body: dict, service: str = "aoss") -> dict:
    """POST a JSON body to <endpoint><path>, signed with SigV4 against `service`.

    For OpenSearch Service domains, pass service='es'. For AOSS, 'aoss'.
    """
    session = boto3.Session()
    credentials = session.get_credentials()
    region = session.region_name
    if region is None:
        raise RuntimeError("AWS region must be set in environment")

    url = endpoint.rstrip("/") + path
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = AWSRequest(method="POST", url=url, data=data, headers=headers)
    SigV4Auth(credentials, service, region).add_auth(req)

    out_headers = dict(req.headers)
    py_req = urllib.request.Request(url, data=data, headers=out_headers, method="POST")
    try:
        with urllib.request.urlopen(py_req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"AOSS request failed: HTTP {e.code} {e.reason}: {body_text}"
        ) from e
```

- [ ] **Step 4: Implement `index.py`**

Create `lambdas/prepare-evaluation/index.py`:

```python
"""Prepare Evaluation Lambda — Stage A.

Reads eval_queries.jsonl from the ProcessBedrockOutput tarball, samples 100
queries stratified by source doc_id, runs kNN against the customer's AOSS
index for each query (top-100), aggregates a candidate pool (~10K unique
docs) and per-query base top-10. Writes three JSONL files to S3 and returns
their URIs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import tarfile
from collections import defaultdict
from urllib.parse import urlparse

import boto3

from aoss_signer import signed_post

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ---------- IO ----------

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
    for line_num, line in enumerate(text.splitlines(), start=1):
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


# ---------- sampling ----------

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


# ---------- AOSS kNN ----------

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


# ---------- handler ----------

def handler(event: dict, context) -> dict:
    logger.info("PrepareEvaluation event: %s", json.dumps({k: v for k, v in event.items() if k != "context"}))

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
    text_field = event.get("text_field", "text")

    s3 = _build_s3_client()

    # 1. Pull eval_queries.jsonl from the upstream tarball
    src_bucket, src_key = parse_s3_uri(eval_queries_s3)
    obj = s3.get_object(Bucket=src_bucket, Key=src_key)
    tarbytes = obj["Body"].read()
    full_pool = extract_eval_queries_from_tarball(tarbytes)
    logger.info("Loaded %d candidate queries from tarball", len(full_pool))

    # 2. Sample 100 stratified
    queries = sample_stratified_queries(full_pool, num_eval, seed)
    logger.info("Sampled %d eval queries", len(queries))

    # 3. kNN per query → pool + base_top10
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

    # 4. Write artifacts
    queries_key = f"{prefix}/queries.jsonl"
    pool_key = f"{prefix}/pool_corpus.jsonl"
    base_key = f"{prefix}/base_top10.jsonl"

    def _put(key: str, lines: list[str]) -> None:
        body = ("\n".join(lines) + "\n").encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/x-ndjson")

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
```

- [ ] **Step 5: Verify tests pass**

```
PYTHONPATH=. pytest test_index.py -v
```
Expected: 9 PASS.

- [ ] **Step 6: Commit**

```bash
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
git add lambdas/prepare-evaluation/
git commit -m "feat(lambda): prepare-evaluation kNN-driven Stage A"
```

---

## Task 4: `evaluate_ft.py` — Stage B (FT embed + rank)

**Files:**
- Create: `training-script/evaluate_ft.py`
- Create: `training-script/tests/test_evaluate_ft.py`

This script:
1. Reads `queries.jsonl` and `pool_corpus.jsonl` from S3 (via `--queries-s3` and `--pool-s3`).
2. Downloads + extracts the train job's `model.tar.gz` from S3 (via `--adapter-s3`) into a temp dir.
3. Loads base SentenceTransformer + applies LoRA adapter via `peft.PeftModel.from_pretrained`.
4. Embeds the pool and the queries, computes top-K within the pool by cosine similarity.
5. Writes `ft_top10.jsonl` to S3 (via `--out-s3`).

We split into testable units:
- `parse_args` — CLI parsing
- `read_jsonl_from_s3(uri)` — IO
- `download_and_extract_adapter(adapter_s3, target_dir)` — IO
- `compute_top_k_rankings(encoder, docs, queries, top_k)` — pure (mockable encoder)
- `main(argv)` — wiring

The encoder loader (`load_finetuned_encoder`) is GPU-bound; tests inject a `MagicMock`.

- [ ] **Step 1: Write failing tests**

Create `training-script/tests/test_evaluate_ft.py`:

```python
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
```

- [ ] **Step 2: Verify tests fail**

```
PYTHONPATH=. pytest tests/test_evaluate_ft.py -v
```
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `evaluate_ft.py`**

Create `training-script/evaluate_ft.py`:

```python
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
import os
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


# ---------- IO ----------

def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"not an s3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _build_s3_client():
    return boto3.client("s3")


def read_jsonl_from_s3(s3, uri: str) -> list[dict]:
    bucket, key = parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


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
        tar.extractall(target_dir)


# ---------- args ----------

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


# ---------- ranking ----------

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


# ---------- production encoder loader ----------

def load_finetuned_encoder(base_model_id: str, adapter_path: str, max_seq_length: int):
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel

    base = AutoModel.from_pretrained(base_model_id, trust_remote_code=False)
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    with tempfile.TemporaryDirectory() as tmp:
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


# ---------- main ----------

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
```

- [ ] **Step 4: Verify tests pass**

```
PYTHONPATH=. pytest tests/test_evaluate_ft.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add training-script/evaluate_ft.py training-script/tests/test_evaluate_ft.py
git commit -m "feat(training): evaluate_ft.py Stage B FT embed + rank"
```

---

## Task 5: `evaluate_judge.py` — Stage C (judge + NDCG + DDB)

**Files:**
- Create: `training-script/evaluate_judge.py`
- Create: `training-script/tests/test_evaluate_judge.py`

Stage C reads four S3 inputs (`queries.jsonl`, `pool_corpus.jsonl`, `base_top10.jsonl`, `ft_top10.jsonl`), unions per-query top-Ks into judge pairs, dispatches Bedrock judging via `bedrock_judge.judge_pairs`, computes graded NDCG@K per config per query, paired bootstrap CI on uplift, then writes `evaluation_report.json` to S3 and the same payload to DynamoDB.

The Bedrock client and DDB resource are factored as `_build_bedrock_client()` / `_build_ddb_resource()` so tests can monkey-patch them. Same pattern as `evaluate_ft.py`'s `_build_s3_client`.

- [ ] **Step 1: Write failing tests**

Create `training-script/tests/test_evaluate_judge.py`:

```python
from __future__ import annotations

import io
import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from evaluate_judge import (
    build_judge_pairs,
    parse_args,
    main,
    _to_dynamodb_safe,
)


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
        # split prompt around 'Document:' and check whether the doc text matches the query intent
        # Heuristic: gold pairs (q1↔d_a, q2↔d_c) get 0.9; everything else 0.1.
        score = "0.1"
        if (("q1" in msg and "d_a" in msg.split("Document:")[1])
                or ("q2" in msg and "d_c" in msg.split("Document:")[1])):
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

    # Report uploaded to S3
    put_calls = [c for c in s3.put_object.call_args_list
                 if c.kwargs["Key"] == "evaluation_report.json"]
    assert len(put_calls) == 1
    report = json.loads(put_calls[0].kwargs["Body"].decode())
    assert report["task_id"] == "test-1"
    metric = report["metrics"]["ndcg_at_2"]
    assert metric["fine_tuned"] > metric["base"]
    assert metric["uplift_mean"] > 0

    # DDB write
    ddb.Table.assert_called_once_with("EvalResults")
    item = table.put_item.call_args.kwargs["Item"]
    assert item["task_id"] == "test-1"
    assert "metrics" in item
```

- [ ] **Step 2: Verify tests fail**

```
PYTHONPATH=. pytest tests/test_evaluate_judge.py -v
```
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `evaluate_judge.py`**

Create `training-script/evaluate_judge.py`:

```python
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


def _build_s3_client():
    return boto3.client("s3")


def _build_bedrock_client():
    return boto3.client("bedrock-runtime")


def _build_ddb_resource():
    return boto3.resource("dynamodb")


def _read_jsonl(s3, uri: str) -> list[dict]:
    bucket, key = parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


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
```

- [ ] **Step 4: Verify tests pass**

```
PYTHONPATH=. pytest tests/test_evaluate_judge.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add training-script/evaluate_judge.py training-script/tests/test_evaluate_judge.py
git commit -m "feat(training): evaluate_judge.py Stage C judge + NDCG + DDB"
```

---

## Task 6: `requirements.txt` + `process_output.py` carry-over from v1

**Files:**
- Modify: `training-script/requirements.txt`
- Modify: `training-script/process_output.py` (only if v1 has not been merged)

- [ ] **Step 1: Update requirements**

Open `training-script/requirements.txt`, ensure the following content:

```
sentence-transformers==5.0.0
transformers==4.44.2
datasets==3.5.0
accelerate==1.6.0
bm25s
faiss-cpu
peft==0.14.0
```

(Adds only `peft==0.14.0` if everything else is already present.)

- [ ] **Step 2: Apply v1 stratified-split if not present**

```
grep -n "stratified_split\|eval-fraction" training-script/process_output.py | head
```

If both appear, **skip the rest of this step** — v1 is already merged.

Otherwise, apply v1 plan Task 2 in full. To recap:
- Add `stratified_split(pairs, eval_fraction, seed)` helper above `parse_args`.
- Add `--eval-fraction` (default 0.0) and `--seed` (default 42) CLI args.
- After `query_doc_pairs` is built (around line 210 of `process_output.py`), call `stratified_split`, write `eval_queries.jsonl` to `args.model_dir`, and feed only the `train_pairs` to the existing BM25 mining loop.

The detailed code for these changes is in v1 plan Task 2.

- [ ] **Step 3: Sanity-check the build**

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
bash build.sh --skip-upload 2>/dev/null || true
tar -tzf build/training-script.tar.gz 2>/dev/null | grep -E "(evaluate_ft|evaluate_judge|eval_metrics|bedrock_judge|peft)"
```

(`build.sh` may or may not support `--skip-upload`; if not, just inspect the resulting tarball locally before pushing.)

Expected: all four new modules are in the tarball.

- [ ] **Step 4: Commit**

```bash
git add training-script/requirements.txt
# also stage process_output.py if you applied v1 changes here
git commit -m "chore(training): peft for LoRA loading; v1 split carry-over"
```

---

## Task 7: `build.sh` — register the new Lambda zip

**Files:**
- Modify: `build.sh`

- [ ] **Step 1: Add the new zip to the upload list**

`build.sh` already automatically packages every `lambdas/*/` subdirectory into `<name>-lambda.zip` (the `for dir in "$SCRIPT_DIR"/lambdas/*/` loop), so the zip will be produced. But the explicit `gh release upload` list at the bottom of the script names each artifact — we need to add `prepare-evaluation-lambda.zip` there.

Open `build.sh`. Find the `gh release upload "$TAG" \` block near the bottom. The current list is:

```bash
gh release upload "$TAG" \
    "$BUILD_DIR/data-extractor-lambda.zip" \
    "$BUILD_DIR/s3-validator-lambda.zip" \
    "$BUILD_DIR/bedrock-orchestrator-lambda.zip" \
    "$BUILD_DIR/register-model-lambda.zip" \
    "$BUILD_DIR/training-script.tar.gz" \
    --repo "$REPO" \
    --clobber
```

Change to:

```bash
gh release upload "$TAG" \
    "$BUILD_DIR/data-extractor-lambda.zip" \
    "$BUILD_DIR/s3-validator-lambda.zip" \
    "$BUILD_DIR/bedrock-orchestrator-lambda.zip" \
    "$BUILD_DIR/register-model-lambda.zip" \
    "$BUILD_DIR/prepare-evaluation-lambda.zip" \
    "$BUILD_DIR/training-script.tar.gz" \
    --repo "$REPO" \
    --clobber
```

- [ ] **Step 2: Test the build locally (no upload)**

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
# Dry-run build only — comment out the gh release lines or use a test tag if needed
bash build.sh test-tag 2>&1 | head -20
ls build/prepare-evaluation-lambda.zip
unzip -l build/prepare-evaluation-lambda.zip | head
```

Expected: `prepare-evaluation-lambda.zip` exists and contains `index.py` + `aoss_signer.py`.

If you can't actually `gh release upload` to a `test-tag`, skip the upload by editing your local copy temporarily, or just verify the zip locally. (No commit needed for this temporary edit.)

- [ ] **Step 3: Commit**

```bash
git add build.sh
git commit -m "build: include prepare-evaluation-lambda.zip in release upload list"
```

---

## Task 8: CFN — DDB + Lambda + IAM + 3 SFN states

**Files:**
- Modify: `opensearch-finetune-poc.yaml`

This is the largest single change in the plan. Five sub-changes, applied in this order:

1. Add `PrepareEvaluationLambdaZipUrl` parameter.
2. Add `EvaluationResultsTable` resource.
3. Add a `DownloadLambdaZips` entry for the new zip.
4. Add `PrepareEvaluationLambda` resource.
5. Augment IAM:
   - `LambdaExecutionRole` (or whichever role `PrepareEvaluationLambda` uses) — `aoss:APIAccessAll` on the customer collection (or arn `*` if you don't pre-bind), `s3:PutObject` and `s3:GetObject` on `DataBucket`.
   - `SageMakerTrainingRole` — `bedrock:InvokeModel` on Claude Haiku model arns + `dynamodb:PutItem` on `EvaluationResultsTable`.
6. Re-route `StartSageMakerTraining` to `PrepareEvaluation`.
7. Insert 3 new SFN states.

- [ ] **Step 1: Add the parameter**

Find the section around line 286 (`DataExtractorLambdaZipUrl`). After `RegisterModelLambdaZipUrl` (around line 301), add:

```yaml
  PrepareEvaluationLambdaZipUrl:
    Type: String
    Default: "https://github.com/zhichao-aws/opensearch-finetuning/releases/download/v1.0.0/prepare-evaluation-lambda.zip"
    Description: "S3/HTTPS URL of the prepare-evaluation Lambda zip."
```

- [ ] **Step 2: Add the DDB table**

Add this resource near the other `AWS::*` resources (a sensible place is just before `SageMakerTrainingRole`):

```yaml
  EvaluationResultsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${ModelName}-EvaluationResults"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: task_id
          AttributeType: S
      KeySchema:
        - AttributeName: task_id
          KeyType: HASH
```

- [ ] **Step 3: Add the DownloadLambdaZips entry**

Find the `DownloadLambdaZips.Properties.ZipUrls` list. Add an entry for the new zip:

```yaml
        - url: !Ref PrepareEvaluationLambdaZipUrl
          s3_key: "lambda-zips/prepare-evaluation.zip"
```

- [ ] **Step 4: Add the Lambda function resource**

Near `RegisterModelLambda`, add a new resource:

```yaml
  PrepareEvaluationLambda:
    Type: AWS::Lambda::Function
    DependsOn: DownloadLambdaZips
    Properties:
      FunctionName: !Sub "${ModelName}-PrepareEvaluation"
      Runtime: python3.12
      Handler: index.handler
      Timeout: 900
      MemorySize: 1024
      Role: !GetAtt FineTuningLambdaInvokeOpenSearchRole.Arn
      Code:
        S3Bucket: !Ref DataBucket
        S3Key: "lambda-zips/prepare-evaluation.zip"
      Environment:
        Variables:
          DATA_BUCKET: !Ref DataBucket
```

The `FineTuningLambdaInvokeOpenSearchRole` already has the OpenSearch / AOSS permissions and the data bucket S3 access (this role is shared with `RegisterModelLambda` and `DataExtractorLambda`). Verify it covers `s3:PutObject` on `DataBucket/*`. If not, add:

```yaml
        - PolicyName: PrepareEvaluationS3Write
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:PutObject
                  - s3:GetObject
                Resource:
                  - !Sub "${DataBucket.Arn}/*"
```

- [ ] **Step 5: Augment SageMakerTrainingRole**

Inside `SageMakerTrainingRole.Properties.Policies`, add two entries:

```yaml
        - PolicyName: BedrockInvokeJudge
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - bedrock:InvokeModel
                Resource:
                  - !Sub "arn:${AWS::Partition}:bedrock:${AWS::Region}::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
                  - !Sub "arn:${AWS::Partition}:bedrock:*::foundation-model/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        - PolicyName: DynamoDbWriteEvaluation
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:PutItem
                Resource: !GetAtt EvaluationResultsTable.Arn
```

- [ ] **Step 6: Allow the SFN role to invoke the new Lambda**

Find the `StateMachineRole` (the IAM role used by the `FineTuneStateMachine`). It already has a list of allowed Lambda ARNs. Add:

```yaml
                  - !GetAtt PrepareEvaluationLambda.Arn
```

to that resource list (alongside the existing `DataExtractorLambda.Arn` etc.).

- [ ] **Step 7: Add eval-fraction hyperparameters to ProcessBedrockOutput** (skip if v1 already applied)

In the `ProcessBedrockOutput` SFN state's `HyperParameters` block, after `"max-corpus-documents.$": "States.Format('{}', $.max_corpus_documents)"`, add:

```json
                      "eval-fraction": "0.2",
                      "seed": "42"
```

- [ ] **Step 8: Re-route `StartSageMakerTraining` next state**

Find:

```yaml
                  "ResultPath": "$.training_job_result",
                  "Next": "CheckDeployEndpoint"
                },
```

(in `StartSageMakerTraining`). Change `"Next"` to `"PrepareEvaluation"`.

- [ ] **Step 9: Insert the three new SFN states**

After the closing `},` of `StartSageMakerTraining` and before `CheckDeployEndpoint`, insert:

```yaml
                "PrepareEvaluation": {
                  "Type": "Task",
                  "Resource": "arn:aws:states:::lambda:invoke",
                  "Comment": "Stage A: sample 100 queries, kNN against AOSS, write artifacts to S3",
                  "Parameters": {
                    "FunctionName": "${PrepareEvaluationLambdaArn}",
                    "Payload": {
                      "task_id.$": "$$.Execution.Name",
                      "model_name.$": "$.model_name",
                      "opensearch_endpoint.$": "$.opensearch_endpoint",
                      "opensearch_index_name.$": "$.index_name",
                      "eval_queries_s3.$": "$.process_output_result.training_data_s3_path",
                      "data_bucket": "${DataBucket}",
                      "data_prefix.$": "States.Format('{}/evaluation', $.model_name)",
                      "num_eval_queries": 100,
                      "pool_size": 100,
                      "top_k": 10,
                      "seed": 42,
                      "text_field": "text"
                    }
                  },
                  "ResultSelector": {
                    "queries_s3.$": "$.Payload.queries_s3",
                    "pool_corpus_s3.$": "$.Payload.pool_corpus_s3",
                    "base_top10_s3.$": "$.Payload.base_top10_s3",
                    "ddb_table_name.$": "$.Payload.ddb_table_name",
                    "candidate_pool_size_unique.$": "$.Payload.candidate_pool_size_unique",
                    "num_eval_queries.$": "$.Payload.num_eval_queries"
                  },
                  "ResultPath": "$.evaluation_prepare_result",
                  "Retry": [
                    {
                      "ErrorEquals": ["Lambda.ServiceException", "Lambda.TooManyRequestsException"],
                      "IntervalSeconds": 2,
                      "MaxAttempts": 3,
                      "BackoffRate": 2
                    }
                  ],
                  "Next": "StartSageMakerEvaluationFTEmbed"
                },
                "StartSageMakerEvaluationFTEmbed": {
                  "Type": "Task",
                  "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                  "Comment": "Stage B: FT model embed pool + queries, write ft_top10.jsonl",
                  "Parameters": {
                    "TrainingJobName.$": "States.Format('eval-ft-{}', $$.Execution.Name)",
                    "AlgorithmSpecification": {
                      "TrainingImage": "${TrainingImageGPU}",
                      "TrainingInputMode": "File"
                    },
                    "OutputDataConfig": {
                      "S3OutputPath.$": "States.Format('s3://${DataBucket}/{}/evaluation/ft-job/', $.model_name)"
                    },
                    "ResourceConfig": {
                      "InstanceCount": 1,
                      "InstanceType.$": "$.config.training_instance_type",
                      "VolumeSizeInGB": 100
                    },
                    "StoppingCondition": {
                      "MaxRuntimeInSeconds": 14400
                    },
                    "RoleArn": "${SageMakerTrainingRoleArn}",
                    "HyperParameters": {
                      "sagemaker_program": "evaluate_ft.py",
                      "sagemaker_submit_directory": "s3://${DataBucket}/training-scripts/training-script.tar.gz",
                      "queries-s3.$": "$.evaluation_prepare_result.queries_s3",
                      "pool-s3.$": "$.evaluation_prepare_result.pool_corpus_s3",
                      "adapter-s3.$": "$.training_job_result.ModelArtifacts.S3ModelArtifacts",
                      "base-model-id.$": "$.base_model_id",
                      "top-k": "10",
                      "max-seq-length.$": "States.Format('{}', $.config.max_seq_length)",
                      "out-s3.$": "States.Format('s3://${DataBucket}/{}/evaluation/ft_top10.jsonl', $.model_name)"
                    },
                    "EnableManagedSpotTraining": false
                  },
                  "ResultSelector": {
                    "ft_top10_s3.$": "States.Format('s3://${DataBucket}/{}/evaluation/ft_top10.jsonl', $.model_name)"
                  },
                  "ResultPath": "$.evaluation_ft_result",
                  "Next": "StartSageMakerEvaluationJudge"
                },
                "StartSageMakerEvaluationJudge": {
                  "Type": "Task",
                  "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                  "Comment": "Stage C: judge pairs via Bedrock, NDCG, write report to DDB + S3",
                  "Parameters": {
                    "TrainingJobName.$": "States.Format('eval-judge-{}', $$.Execution.Name)",
                    "AlgorithmSpecification": {
                      "TrainingImage": "${TrainingImageGPU}",
                      "TrainingInputMode": "File"
                    },
                    "OutputDataConfig": {
                      "S3OutputPath.$": "States.Format('s3://${DataBucket}/{}/evaluation/judge-job/', $.model_name)"
                    },
                    "ResourceConfig": {
                      "InstanceCount": 1,
                      "InstanceType": "ml.m5.xlarge",
                      "VolumeSizeInGB": 30
                    },
                    "StoppingCondition": {
                      "MaxRuntimeInSeconds": 7200
                    },
                    "RoleArn": "${SageMakerTrainingRoleArn}",
                    "HyperParameters": {
                      "sagemaker_program": "evaluate_judge.py",
                      "sagemaker_submit_directory": "s3://${DataBucket}/training-scripts/training-script.tar.gz",
                      "queries-s3.$": "$.evaluation_prepare_result.queries_s3",
                      "pool-s3.$": "$.evaluation_prepare_result.pool_corpus_s3",
                      "base-top10-s3.$": "$.evaluation_prepare_result.base_top10_s3",
                      "ft-top10-s3.$": "$.evaluation_ft_result.ft_top10_s3",
                      "judge-model-id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                      "judge-concurrency": "16",
                      "top-k": "10",
                      "bootstrap-resamples": "1000",
                      "seed": "42",
                      "ddb-table-name.$": "$.evaluation_prepare_result.ddb_table_name",
                      "task-id.$": "$$.Execution.Name",
                      "out-s3.$": "States.Format('s3://${DataBucket}/{}/evaluation/evaluation_report.json', $.model_name)",
                      "base-model-id.$": "$.base_model_id"
                    },
                    "EnableManagedSpotTraining": false
                  },
                  "ResultPath": "$.evaluation_judge_result",
                  "Next": "CheckDeployEndpoint"
                },
```

- [ ] **Step 10: Wire the new Lambda ARN into `Fn::Sub` mapping**

The state-machine `DefinitionString` uses `${PrepareEvaluationLambdaArn}` interpolation. Find the `Fn::Sub` mapping at the end of the state machine definition (around line 1548) where other lambda ARNs are mapped:

```yaml
            BedrockOrchestratorLambdaArn: !GetAtt BedrockOrchestratorLambda.Arn
```

Add an entry:

```yaml
            PrepareEvaluationLambdaArn: !GetAtt PrepareEvaluationLambda.Arn
```

- [ ] **Step 11: Validate the template parses**

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
aws cloudformation validate-template \
  --template-body file://opensearch-finetune-poc.yaml \
  --region us-east-1 | head
```

Expected: returns the template description without errors.

- [ ] **Step 12: Commit**

```bash
git add opensearch-finetune-poc.yaml
git commit -m "feat(cfn): EvaluationResults DDB + 3-stage evaluation SFN"
```

---

## Task 9: End-to-end smoke test in dev account

Manual integration check.

- [ ] **Step 1: Build & upload artifacts**

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning
./build.sh
```

Expected: build succeeds; the GitHub release tag has all 6 artifacts including `prepare-evaluation-lambda.zip`.

- [ ] **Step 2: Deploy stack to dev account**

```
export STACK=ft-eval-v3-smoke-$(date +%s)
export REGION=us-west-2
export BUCKET=<your-dev-bucket>
export AOSS_ENDPOINT=https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com
export AOSS_INDEX=test-ecs-v11

# Upload the template you just built (or reuse the released one)
aws s3 cp opensearch-finetune-poc.yaml s3://$BUCKET/cfn/opensearch-finetune-poc.yaml --region $REGION

aws cloudformation create-stack \
  --region $REGION --stack-name $STACK \
  --template-url https://$BUCKET.s3.$REGION.amazonaws.com/cfn/opensearch-finetune-poc.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=ModelName,ParameterValue=eval-v3-smoke \
    ParameterKey=InputType,ParameterValue=s3 \
    ParameterKey=S3CorpusPath,ParameterValue=s3://$BUCKET/test/corpus.jsonl \
    ParameterKey=DeployEndpoint,ParameterValue=false \
    ParameterKey=RegisterConnector,ParameterValue=false \
    ParameterKey=MaxSteps,ParameterValue=2 \
    ParameterKey=MaxQueryDocuments,ParameterValue=200 \
    ParameterKey=OpenSearchEndpoint,ParameterValue=$AOSS_ENDPOINT
```

Note: `S3CorpusPath` must be **the same docs that exist in the AOSS index** — otherwise base kNN will return doc_ids that aren't in the corpus dump. For the smoke test, the easiest way is to dump `test-ecs-v11`'s docs to `s3://$BUCKET/test/corpus.jsonl` (each line `{"text": ...}`) before triggering.

- [ ] **Step 3: Trigger an execution**

```
aws stepfunctions start-execution \
  --region $REGION \
  --state-machine-arn $(aws cloudformation describe-stacks \
      --stack-name $STACK --region $REGION \
      --query "Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue" \
      --output text) \
  --name v3-smoke-$(date +%s) \
  --input "$(cat <<EOF
{
  "input_type": "s3",
  "model_name": "eval-v3-smoke",
  "base_model_id": "BAAI/bge-m3",
  "deploy_endpoint": "false",
  "register_connector": "false",
  "max_corpus_documents": "1000",
  "max_query_documents": "200",
  "s3_corpus_path": "s3://$BUCKET/test/corpus.jsonl",
  "opensearch_endpoint": "$AOSS_ENDPOINT",
  "index_name": "$AOSS_INDEX",
  "config": {}
}
EOF
)"
```

- [ ] **Step 4: After Success, inspect the DDB row**

```
aws dynamodb get-item --region $REGION \
  --table-name eval-v3-smoke-EvaluationResults \
  --key '{"task_id": {"S": "<the execution name>"}}'
```

Expected: returns the report payload.

- [ ] **Step 5: Inspect the S3 report file**

```
aws s3 cp s3://$BUCKET/eval-v3-smoke/evaluation/evaluation_report.json - \
  | python -m json.tool
```

Expected: same payload, with `metrics.ndcg_at_10` containing all 5 keys, `judge_pairs_count` ≤ 2 000, `num_eval_queries` ≤ 100.

- [ ] **Step 6: Sanity-check metrics**

With `MaxSteps=2`, training is too short to produce a real signal. What you should see:
- `base` and `fine_tuned` NDCG@10 in `[0, 1]`.
- `uplift_ci_low ≤ uplift_mean ≤ uplift_ci_high`.

If the workflow fails at `PrepareEvaluation`, check CloudWatch logs for `eval-v3-smoke-PrepareEvaluation` — most likely cause is AOSS access denial (IAM role not mapped to OpenSearch backend role) or doc-id mismatch (corpus.jsonl != AOSS index contents).

- [ ] **Step 7: Tear down**

```
aws cloudformation delete-stack --stack-name $STACK --region $REGION
```

---

## Task 10: Self-review + portability checklist

This is a documentation step; nothing to commit.

- [ ] **Step 1: Run the full test suite**

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning/training-script
PYTHONPATH=. pytest tests/ -v
```
Expected: ~30 passing.

```
cd /workplace/qianheng/FineTunePOC/opensearch-finetuning/lambdas/prepare-evaluation
PYTHONPATH=. pytest test_index.py -v
```
Expected: 9 passing.

- [ ] **Step 2: Verify portability invariants**

These are the design constraints that let Stage C run on ECS in prod with **zero code changes**:

1. `evaluate_judge.py` does not import any `sagemaker.*` modules and does not read from `SM_CHANNEL_*` env vars. ✓ (uses only `boto3` + `--*-s3` args)
2. `evaluate_ft.py` likewise. ✓
3. `bedrock_judge.py` and `eval_metrics.py` are pure logic, no SM dependencies. ✓
4. All inter-stage data flows through explicit S3 URIs. ✓
5. The Bedrock client and DDB resource are factored as `_build_*()` factories so prod can override them via dependency injection if needed. ✓ (already used in tests)

When prod migrates Stage C to ECS in `/workplace/qianheng/AOS/src/`:
- Copy `training-script/{evaluate_judge,bedrock_judge,eval_metrics}.py` to a new ECS package directory.
- Wrap with the existing ECS `entrypoint.py` registry pattern (see `AWSSearchServiceModelFineTuneECSTasks/.../entrypoint.py`).
- Add the equivalent IAM policies (`bedrock:InvokeModel` for the judge model + `dynamodb:PutItem`).
- Replace SFN `sagemaker:createTrainingJob.sync` with `ecs:runTask.sync` for the judge stage.
- No test changes — the unit tests stay portable.

Stage A's Lambda likewise has no POC-specific dependencies; it ports as a Lambda function in prod's CDK if desired, or wraps in an ECS entrypoint for consistency with prod's existing fine-tune pattern.

Stage B is genuinely SM-bound (LoRA + sentence-transformers + GPU). Prod will keep this as a SageMaker training job invocation.

---

## Post-Phase-1 Follow-ups (out of scope here)

- Multi-run averaging on the LLM judge (HLD §5.3 N=3).
- FT-side pool union (close the pool-eval bias toward base).
- Migrate Stages A and C to ECS in `/workplace/qianheng/AOS/src/` (ECS already used there for fine-tune).
- Console UI / API to read DDB rows by `task_id`.
- Hold-out semantics — eval queries strictly outside the training set.
