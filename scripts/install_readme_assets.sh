#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "python3 not found; install Python 3.11+ and retry." >&2
  exit 1
fi

"$PY" -m pip install -q -r "$ROOT/scripts/requirements-readme-assets.txt"
"$PY" "$ROOT/scripts/gen_readme_png.py"
