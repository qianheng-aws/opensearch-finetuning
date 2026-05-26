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
