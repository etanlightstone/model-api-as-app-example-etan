#!/usr/bin/env bash
#
# Domino App entrypoint for the Model-API-as-App harness.
#
# Domino runs this script to start a published App. The platform expects the
# web server to listen on port 8888 and serves it behind a reverse proxy at
# `https://<deployment>/<owner>/<project>/r/.../<app>/...`. FastAPI + Starlette
# build absolute URLs from the forwarded headers uvicorn's proxy-headers mode
# trusts, so the self-doc UI and curl snippets render the externally-reachable
# URL rather than `localhost:8888`.
set -euo pipefail

cd "$(dirname "$0")"

# Domino apps must bind 0.0.0.0:8888.
export APP_PORT="${APP_PORT:-8888}"

exec uvicorn app:app \
  --host 0.0.0.0 \
  --port "${APP_PORT}" \
  --proxy-headers \
  --forwarded-allow-ips '*' \
  --workers 1
