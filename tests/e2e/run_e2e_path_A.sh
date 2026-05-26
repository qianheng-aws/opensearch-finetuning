#!/usr/bin/env bash
# End-to-end Path A test: deploy + invoke Lambda → SM Stage B → SM Stage C.
# Idempotent (re-runnable). See README.md for prereqs.

set -euo pipefail

REGION="${REGION:-us-west-2}"
ACCOUNT="${ACCOUNT:-330700426359}"
RUN_PREFIX="${RUN_PREFIX:-qh-eval-poc-A}"
MODEL_NAME="${MODEL_NAME:-qh-eval-A}"
DDB_TABLE="${DDB_TABLE:-${MODEL_NAME}-EvaluationResults}"
LAMBDA_NAME="${LAMBDA_NAME:-${RUN_PREFIX}-PrepareEvaluation}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-${RUN_PREFIX}-LambdaRole}"
SM_ROLE_NAME="${SM_ROLE_NAME:-AmazonSageMaker-ExecutionRole-20260126T154349}"
AOSS_ENDPOINT="${AOSS_ENDPOINT:-https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com}"
AOSS_INDEX="${AOSS_INDEX:-ms-marco}"
LORA_TARBALL_SRC="${LORA_TARBALL_SRC:-s3://yuanchu-data/sagemaker/integ-test-data/2026-05-21/03_final/train_output/model.tar.gz}"

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/eval-poc-A-$$}"
mkdir -p "$WORK_DIR"
echo "==> Working dir: $WORK_DIR"
echo "==> Run prefix: $RUN_PREFIX"
echo "==> Model name: $MODEL_NAME"
echo

#############################################
# Step 0: Build & upload training-script.tar.gz
#############################################
echo "==> Step 0: training-script.tar.gz"
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='tests' \
    -czf "$WORK_DIR/training-script.tar.gz" -C "$REPO_ROOT/training-script" .
aws s3 cp "$WORK_DIR/training-script.tar.gz" "s3://yuanchu-data/$RUN_PREFIX/training-script.tar.gz" --region "$REGION"
echo

#############################################
# Step 1: Build the prepare-evaluation Lambda zip
#############################################
echo "==> Step 1: Build Lambda zip"
LAMBDA_DIR="$REPO_ROOT/lambdas/prepare-evaluation"
STAGE="$WORK_DIR/lambda-stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' \
      --exclude='test_*.py' --exclude='*.pyc' "$LAMBDA_DIR"/ "$STAGE"/
pip install --quiet --target "$STAGE" -r "$LAMBDA_DIR/requirements.txt"
( cd "$STAGE" && zip -rq "$WORK_DIR/prepare-evaluation-lambda.zip" . -x '*.pyc' '__pycache__/*' )
rm -rf "$STAGE"
echo

#############################################
# Step 2: Create Lambda IAM role (idempotent)
#############################################
echo "==> Step 2: Lambda IAM role"
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" >/dev/null 2>&1; then
    cat > "$WORK_DIR/trust.json" <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
    aws iam create-role --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document file://$WORK_DIR/trust.json >/dev/null
    aws iam attach-role-policy --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi

cat > "$WORK_DIR/lambda-extra.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"AOSSAccess","Effect":"Allow","Action":"aoss:APIAccessAll",
     "Resource":"arn:aws:aoss:$REGION:$ACCOUNT:collection/5l2oxy87av1hbwupnpoc"},
    {"Sid":"S3Yuanchu","Effect":"Allow",
     "Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
     "Resource":["arn:aws:s3:::yuanchu-data","arn:aws:s3:::yuanchu-data/*"]}
  ]
}
EOF
aws iam put-role-policy --role-name "$LAMBDA_ROLE_NAME" --policy-name extra \
    --policy-document file://$WORK_DIR/lambda-extra.json
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name "$LAMBDA_ROLE_NAME" --query 'Role.Arn' --output text)
echo "  Role: $LAMBDA_ROLE_ARN"
sleep 5  # IAM propagation
echo

#############################################
# Step 3: Add Lambda role to AOSS data access policy (idempotent)
#############################################
echo "==> Step 3: AOSS data access policy"
P=$(aws opensearchserverless get-access-policy --type data --name easy-ase-test-collection --region "$REGION")
VER=$(echo "$P" | python3 -c "import json,sys;print(json.load(sys.stdin)['accessPolicyDetail']['policyVersion'])")
echo "$P" | python3 -c "
import json, sys
d = json.load(sys.stdin)['accessPolicyDetail']['policy']
ME = '$LAMBDA_ROLE_ARN'
changed = False
for r in d:
    if 'Easy' in r.get('Description', ''):
        if ME not in r['Principal']:
            r['Principal'].append(ME)
            changed = True
            print('Added principal', file=sys.stderr)
        else:
            print('Principal already in policy', file=sys.stderr)
with open('$WORK_DIR/policy_new.json', 'w') as f:
    json.dump(d, f)
sys.exit(0 if changed else 100)
" && {
    aws opensearchserverless update-access-policy --region "$REGION" \
        --type data --name easy-ase-test-collection \
        --policy-version "$VER" \
        --policy file://$WORK_DIR/policy_new.json >/dev/null
    echo "  Policy updated, sleeping 30s for propagation"
    sleep 30
} || echo "  Policy already up to date"
echo

#############################################
# Step 4: Deploy or update the Lambda
#############################################
echo "==> Step 4: Lambda deploy"
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --region "$REGION" \
        --function-name "$LAMBDA_NAME" \
        --zip-file fileb://$WORK_DIR/prepare-evaluation-lambda.zip >/dev/null
else
    aws lambda create-function --region "$REGION" \
        --function-name "$LAMBDA_NAME" \
        --runtime python3.12 --handler index.handler \
        --memory-size 1024 --timeout 900 \
        --role "$LAMBDA_ROLE_ARN" \
        --zip-file fileb://$WORK_DIR/prepare-evaluation-lambda.zip \
        --environment Variables={DATA_BUCKET=yuanchu-data} >/dev/null
fi
sleep 5  # function update settle
echo

#############################################
# Step 5: Build eval_queries tarball + upload
#############################################
echo "==> Step 5: eval_queries.jsonl tarball"
python3 - <<'PYEOF' > "$WORK_DIR/eval_queries.jsonl"
import json
with open("eval_queries_100.txt" if __import__("os").path.exists("eval_queries_100.txt") else "/dev/stdin") as f:
    for i, q in enumerate((l.strip() for l in f if l.strip()), 1):
        print(json.dumps({"query": q, "doc_id": f"msmarco-{i:03d}"}))
PYEOF
echo '{"anchor":"placeholder","positive":"placeholder","negatives":[]}' > "$WORK_DIR/training_data.jsonl"
( cd "$WORK_DIR" && tar czf model.tar.gz eval_queries.jsonl training_data.jsonl )
aws s3 cp "$WORK_DIR/model.tar.gz" "s3://yuanchu-data/$RUN_PREFIX/process-output/model.tar.gz" --region "$REGION"
echo

#############################################
# Step 6: Invoke Lambda (Stage A)
#############################################
echo "==> Step 6: Stage A — Lambda invoke"
TASK_ID="$RUN_PREFIX-stage-a-$(date +%s)"
sed -e "s|{TASK_ID}|$TASK_ID|g" -e "s|{MODEL_NAME}|$MODEL_NAME|g" \
    -e "s|{RUN_PREFIX}|$RUN_PREFIX|g" event_template.json > "$WORK_DIR/event.json"

aws lambda invoke --region "$REGION" \
    --function-name "$LAMBDA_NAME" \
    --cli-binary-format raw-in-base64-out \
    --payload file://$WORK_DIR/event.json \
    "$WORK_DIR/lambda-output.json" >/dev/null
echo "  Lambda response:"
cat "$WORK_DIR/lambda-output.json" | python3 -m json.tool | sed 's/^/    /'
echo

#############################################
# Step 7: Upload LoRA adapter (idempotent)
#############################################
echo "==> Step 7: LoRA adapter"
aws s3 cp "$LORA_TARBALL_SRC" "s3://yuanchu-data/$RUN_PREFIX/lora-adapter.tar.gz" --region "$REGION"
echo

#############################################
# Step 8: Submit Stage B SM job
#############################################
echo "==> Step 8: Stage B — SM GPU embed"
STAGE_B_JOB="$RUN_PREFIX-ft-$(date +%s)"
sed -e "s|{JOB_NAME}|$STAGE_B_JOB|g" -e "s|{RUN_PREFIX}|$RUN_PREFIX|g" \
    sm_stage_b_template.json > "$WORK_DIR/sm-stage-b.json"
aws sagemaker create-training-job --region "$REGION" \
    --cli-input-json file://$WORK_DIR/sm-stage-b.json | python3 -c "
import json, sys
print('  ARN:', json.load(sys.stdin)['TrainingJobArn'])
"

# Poll until done
echo -n "  polling: "
while :; do
    OUT=$(aws sagemaker describe-training-job --training-job-name "$STAGE_B_JOB" \
        --region "$REGION" --query "{S:TrainingJobStatus,Sec:TrainingTimeInSeconds}" --output json)
    STATUS=$(echo "$OUT" | python3 -c "import json,sys;print(json.load(sys.stdin)['S'])")
    SEC=$(echo "$OUT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('Sec',''))")
    echo -n "[$STATUS:$SEC] "
    if [ "$STATUS" != "InProgress" ]; then echo; break; fi
    sleep 30
done
if [ "$STATUS" != "Completed" ]; then
    echo "ERROR Stage B failed: $STATUS"
    aws sagemaker describe-training-job --training-job-name "$STAGE_B_JOB" --region "$REGION" \
        --query "FailureReason" --output text
    exit 1
fi
echo

#############################################
# Step 9: Create DDB table (idempotent) + extend SM IAM
#############################################
echo "==> Step 9: DDB table + SM role extension"
if ! aws dynamodb describe-table --table-name "$DDB_TABLE" --region "$REGION" >/dev/null 2>&1; then
    aws dynamodb create-table --region "$REGION" \
        --table-name "$DDB_TABLE" \
        --billing-mode PAY_PER_REQUEST \
        --attribute-definitions AttributeName=task_id,AttributeType=S \
        --key-schema AttributeName=task_id,KeyType=HASH >/dev/null
    aws dynamodb wait table-exists --table-name "$DDB_TABLE" --region "$REGION"
fi

cat > "$WORK_DIR/sm-extra.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"BedrockJudgeInvoke","Effect":"Allow","Action":"bedrock:InvokeModel",
     "Resource":[
       "arn:aws:bedrock:$REGION::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
       "arn:aws:bedrock:*::foundation-model/us.anthropic.claude-haiku-4-5-20251001-v1:0",
       "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
     ]},
    {"Sid":"DDBPutEvalResults","Effect":"Allow","Action":"dynamodb:PutItem",
     "Resource":"arn:aws:dynamodb:$REGION:$ACCOUNT:table/$DDB_TABLE"},
    {"Sid":"S3Yuanchu","Effect":"Allow",
     "Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
     "Resource":["arn:aws:s3:::yuanchu-data","arn:aws:s3:::yuanchu-data/*"]}
  ]
}
EOF
aws iam put-role-policy --role-name "$SM_ROLE_NAME" --policy-name eval-poc-extra \
    --policy-document file://$WORK_DIR/sm-extra.json
echo

#############################################
# Step 10: Submit Stage C SM job
#############################################
echo "==> Step 10: Stage C — SM CPU judge"
STAGE_C_JOB="$RUN_PREFIX-judge-$(date +%s)"
sed -e "s|{JOB_NAME}|$STAGE_C_JOB|g" \
    -e "s|{RUN_PREFIX}|$RUN_PREFIX|g" \
    -e "s|{DDB_TABLE}|$DDB_TABLE|g" \
    sm_stage_c_template.json > "$WORK_DIR/sm-stage-c.json"
aws sagemaker create-training-job --region "$REGION" \
    --cli-input-json file://$WORK_DIR/sm-stage-c.json | python3 -c "
import json, sys
print('  ARN:', json.load(sys.stdin)['TrainingJobArn'])
"

echo -n "  polling: "
while :; do
    OUT=$(aws sagemaker describe-training-job --training-job-name "$STAGE_C_JOB" \
        --region "$REGION" --query "{S:TrainingJobStatus,Sec:TrainingTimeInSeconds}" --output json)
    STATUS=$(echo "$OUT" | python3 -c "import json,sys;print(json.load(sys.stdin)['S'])")
    SEC=$(echo "$OUT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('Sec',''))")
    echo -n "[$STATUS:$SEC] "
    if [ "$STATUS" != "InProgress" ]; then echo; break; fi
    sleep 30
done
if [ "$STATUS" != "Completed" ]; then
    echo "ERROR Stage C failed: $STATUS"
    aws sagemaker describe-training-job --training-job-name "$STAGE_C_JOB" --region "$REGION" \
        --query "FailureReason" --output text
    exit 1
fi
echo

#############################################
# Step 11: Print results
#############################################
echo "============================================="
echo "  E2E COMPLETE"
echo "============================================="
echo
echo "=== evaluation_report.json ==="
aws s3 cp "s3://yuanchu-data/$RUN_PREFIX/run-1/evaluation_report.json" - --region "$REGION" \
    | python3 -m json.tool
echo
echo "=== DDB row (task_id=$STAGE_C_JOB) ==="
aws dynamodb get-item --region "$REGION" \
    --table-name "$DDB_TABLE" \
    --key "{\"task_id\":{\"S\":\"$STAGE_C_JOB\"}}" \
    --query "Item.metrics" --output json
echo
echo "Working dir kept at: $WORK_DIR (clean up with rm -rf if desired)"
