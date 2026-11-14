#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PORT="${1:-7860}"
PYTHON="${PYTHON:-python3}"

echo "Starting UbaNova Web Server on http://127.0.0.1:$PORT"
exec "$PYTHON" server.py --port "$PORT"
