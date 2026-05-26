"""SigV4-signed AOSS request helper, matching the pattern used in
lambdas/data-extractor/index.py and refresh_embedding_ecs/tests/setup_test_index.py.

We use requests + requests_aws4auth here (NOT urllib + botocore SigV4Auth)
because AOSS rejects requests signed by botocore.SigV4Auth with HTTP 403
when a body is sent — likely a header / hash mismatch in how botocore
serializes the request to urllib. requests_aws4auth handles AOSS correctly
across all data-plane operations.

Both `requests` and `requests_aws4auth` are bundled into the Lambda zip
by `build.sh` (they're not in the Lambda Python runtime by default).
"""

from __future__ import annotations

import boto3
import requests
from requests_aws4auth import AWS4Auth


def signed_post(endpoint: str, path: str, body: dict, service: str = "aoss") -> dict:
    """POST a JSON body to <endpoint><path>, signed with SigV4 against `service`.

    For OpenSearch Service domains, pass service='es'. For AOSS, 'aoss'.
    """
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError("No AWS credentials available")
    creds = creds.get_frozen_credentials()
    region = session.region_name
    if region is None:
        raise RuntimeError("AWS region must be set in environment")

    auth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        region,
        service,
        session_token=creds.token,
    )

    url = endpoint.rstrip("/") + path
    resp = requests.post(
        url,
        auth=auth,
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"AOSS request failed: HTTP {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()
