"""Call the model (hosted as a Domino App) from off-platform using a PAT.

Inside a Domino workload you'd fetch a short-lived token from the local proxy
(`curl -s http://localhost:8899/access-token`). From a laptop or CI there's no
proxy, so we authenticate with a Domino **Personal Access Token** passed straight
through as the bearer token — the same header the in-workload token uses.

Usage:
    export DOMINO_PAT="<your Domino personal access token>"
    python simple_with_pat.py
"""

from __future__ import annotations

import os
import sys

import requests

# The model's sync endpoint. Override per deployment via MODEL_API_URL if needed.
MODEL_API_URL = os.environ.get(
    "MODEL_API_URL",
    "https://apps.cloud-dogfood.domino.tech/apps/91b27ca1-7996-4b7a-b966-e99b30b9cc0e/models/weathclasser/latest/model",
)

# The input record. Shape must match the model's schema (see the app's Endpoints
# page for the exact fields and an example payload).
PAYLOAD = {
    "data": {
        "month": "0",
        "week_of": "0",
        "state": "example",
        "precipitation": "0.0",
        "wind_speed": "0.0",
        "wind_direction": "0.0",
    }
}


def main() -> int:
    pat = os.environ.get("DOMINO_PAT")
    if not pat:
        print(
            "Set DOMINO_PAT to a Domino Personal Access Token first, e.g.\n"
            '  export DOMINO_PAT="<your token>"',
            file=sys.stderr,
        )
        return 1

    resp = requests.post(
        MODEL_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {pat}",
        },
        json=PAYLOAD,
        timeout=30,
    )

    if not resp.ok:
        print(f"Request failed: {resp.status_code} {resp.reason}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        return 1

    print(resp.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
