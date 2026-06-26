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
import re
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


def format_error(resp: requests.Response) -> str:
    """Render a server error for humans, with guidance for schema mismatches."""
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text

    out = [f"Request failed: HTTP {resp.status_code} {resp.reason}"]
    if resp.status_code == 422 and isinstance(detail, str) and "validation error" in detail.lower():
        missing = re.findall(r"^(\S.+)\n\s+Field required", detail, flags=re.MULTILINE)
        out += ["", "The payload doesn't match this model's expected input schema."]
        out.append(f"  You sent:         {', '.join(PAYLOAD['data'].keys())}")
        if missing:
            out.append(f"  Missing required: {', '.join(missing)}")
        out += [
            "",
            "What to do:",
            "  1. Open the app in a browser and check its Endpoints page — it lists this",
            "     model's exact fields and a ready-to-copy example payload.",
            "  2. Update PAYLOAD at the top of this script to match, then re-run.",
        ]
    else:
        out.append(str(detail))
    return "\n".join(out)


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
        print(format_error(resp), file=sys.stderr)
        return 1

    print(resp.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
