# Evaluation Pipeline End-to-End Tests

Reproducible scripts for the v3 evaluation pipeline (3-stage: Lambda → SM GPU → SM CPU).

## What's here

| File | Purpose |
|---|---|
| `run_e2e_path_A.sh` | Full Path A: deploy + invoke Lambda Stage A → SM Stage B → SM Stage C, all 3 stages on real AWS. Idempotent (re-runnable). |
| `run_local_stage_ab.py` | Path C variant: skip cloud Lambda + GPU; run base + LoRA-merged FT encoders locally; produce 4 S3 artifacts. Useful for fast iteration. |
| `eval_queries_100.txt` | The 100 ms-marco-style queries used for Path A smoke. |
| `event_template.json` | Lambda invoke event payload template (substitute `{TASK_ID}`, `{MODEL_NAME}`, `{RUN_PREFIX}`). |
| `sm_stage_b_template.json` / `sm_stage_c_template.json` | SageMaker `create-training-job` request body templates. |

## Path A: full cloud e2e

Prereq:

- AWS profile `default` set up via `ada credentials process` (Admin role on the test account).
- AOSS collection `5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com` (`ase-test-collection`)
  with `ms-marco` index using `embedding_v9..v13` 768-dim fields and `test-reindex-search-pipeline`
  configured with `default_model_id=702cbcf5-...` (gte-base).
- The IAM principal you assume (e.g. `arn:aws:sts::ACCT:assumed-role/Admin/...`) must be in
  the `easy-ase-test-collection` AOSS data access policy. If not, see
  `add_principal_to_aoss_policy.sh`.
- yuanchu LoRA adapter at
  `s3://yuanchu-data/sagemaker/integ-test-data/2026-05-21/03_final/train_output/model.tar.gz`.

Run:

```bash
cd tests/e2e
./run_e2e_path_A.sh
```

The script does, in order:

1. Build & upload `training-script.tar.gz` to S3.
2. Build & deploy `prepare-evaluation` Lambda (creates IAM role, attaches inline policy,
   adds the role to the AOSS data access policy).
3. Build the eval_queries tarball from `eval_queries_100.txt` and upload.
4. Invoke the Lambda → produces `queries.jsonl`, `pool_corpus.jsonl`, `base_top10.jsonl`.
5. Submit Stage B (`evaluate_ft.py`) SM training job (GPU `ml.g5.2xlarge`, ~5 min).
6. Submit Stage C (`evaluate_judge.py`) SM training job (CPU `ml.m5.xlarge`, ~7 min).
7. Print final NDCG numbers + DDB row.

Expected outcome:

- 100 queries
- ~9876 unique candidate docs in pool
- ~1700–2000 (q, doc) Bedrock Claude Haiku judge calls
- NDCG@10 base/fine_tuned printed at end
- Row written to `qh-eval-A-EvaluationResults` DDB table

Total wall-clock: ~20-25 min (most of it is SageMaker container startup; actual compute
is ~3 min embed + ~5 min judging).

Total cost: ~$0.5 GPU + ~$0.3 CPU + ~$2 Bedrock judge ≈ $3.

## Path C: fast local Stage A+B

Useful when you want to iterate on `evaluate_judge.py` without paying for GPU time:

```bash
pip install sentence-transformers==5.0.0 transformers==4.44.2 peft==0.14.0 torch
python run_local_stage_ab.py \
    --base-model-id Alibaba-NLP/gte-multilingual-base \
    --corpus s3://yuanchu-data/sagemaker/integ-test-data/2026-05-21/01_inputs/hnm_corpus/corpus.jsonl \
    --queries s3://yuanchu-data/sagemaker/integ-test-data/2026-05-21/01_inputs/hnm_queries/queries.jsonl \
    --adapter-tarball s3://yuanchu-data/sagemaker/integ-test-data/2026-05-21/03_final/train_output/model.tar.gz \
    --out-prefix s3://yuanchu-data/qh-eval-poc-c/run-1
```

Then submit Stage C the same way as Path A step 6.

## Bugs found during e2e bring-up (committed on this branch)

1. `cc396b6` — `boto3.client(..., region_name=...)` required in SM containers.
2. `d393fb9` — `aoss_signer.signed_post` must use `requests` + `requests_aws4auth` (not
   urllib + botocore SigV4Auth — AOSS rejects the latter with 403).
3. `cbe57df` — NDJSON readers must use `text.split("\n")`, not `str.splitlines()`.
   The latter splits on Unicode line separators (U+2028, U+2029, U+0085) that legitimately
   appear inside ms-marco doc text but are not record boundaries.
4. `cc4c4dd` — `evaluate_ft.py` must call `AutoModel.from_pretrained(..., trust_remote_code=True)`
   for models with custom modeling code (e.g. Alibaba-NLP/gte-multilingual-base).
