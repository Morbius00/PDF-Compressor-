#!/usr/bin/env bash
# Starts Clyro and opens it in your browser. First run sets up its own environment.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up (one time)…"
  python3 -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi

PORT="${PORT:-5050}"
( sleep 2 && open "http://127.0.0.1:${PORT}" 2>/dev/null || true ) &
PORT="$PORT" exec .venv/bin/python -W ignore app.py
