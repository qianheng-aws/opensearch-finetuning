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
