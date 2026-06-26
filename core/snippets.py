"""Generate the ready-to-paste, all-in-one curl snippets (§5.7).

Because the endpoints rely on Domino app auth, a programmatic caller just needs
a valid Domino access token in the ``Authorization`` header. Inside any Domino
workload that token is available ephemerally from the local token proxy, so the
snippet fetches it inline each run (no static key, no stale-token footgun) and
falls back to a pasted Personal Access Token for off-platform callers.
"""

from __future__ import annotations

import json

from core import links, settings
from core.schema import Schema, example_record


def _payload_block(record: dict) -> str:
    return json.dumps(record, indent=2)


def sync_curl(base: str, slug: str, schema: Schema, *, in_workload: bool = True) -> str:
    record = example_record(schema)
    body = json.dumps({"data": record})
    url = links.sync_url(base, slug)
    if in_workload:
        token_line = f'TOKEN=$(curl -s {settings.TOKEN_PROXY_URL})'
    else:
        token_line = 'TOKEN="<paste a Domino Personal Access Token>"'
    return (
        f"{token_line}\n\n"
        f'curl -X POST "{url}" \\\n'
        f"  -H 'Content-Type: application/json' \\\n"
        f'  -H "Authorization: Bearer $TOKEN" \\\n'
        f"  -d '{body}'"
    )


def async_curl(base: str, slug: str, schema: Schema, *, in_workload: bool = True) -> str:
    record = example_record(schema)
    body = json.dumps({"parameters": record})
    abase = links.async_base(base, slug)
    if in_workload:
        token_line = f'TOKEN=$(curl -s {settings.TOKEN_PROXY_URL})'
    else:
        token_line = 'TOKEN="<paste a Domino Personal Access Token>"'
    if in_workload:
        poll_auth = f'"Authorization: Bearer $(curl -s {settings.TOKEN_PROXY_URL})"'
    else:
        poll_auth = '"Authorization: Bearer $TOKEN"'
    return (
        f"{token_line}\n"
        f'BASE="{abase}"\n\n'
        f'ID=$(curl -s -X POST "$BASE" \\\n'
        f"  -H 'Content-Type: application/json' -H \"Authorization: Bearer $TOKEN\" \\\n"
        f"  -d '{body}' | jq -r .asyncPredictionId)\n\n"
        f'curl -s "$BASE/$ID" -H {poll_auth}   # repeat until status is terminal'
    )
