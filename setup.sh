#!/usr/bin/env bash
# setup.sh — one-shot setup + launch for relational-reality.
#
# Creates a local virtual environment (.venv), installs the required packages
# into it, and then launches the project. One command and you're running:
#
#   ./setup.sh
#
# Pass --no-run to only set up the environment without starting the app
# (e.g. when you just want to refresh dependencies):
#
#   ./setup.sh --no-run
#
# Safe to re-run; it reuses the existing venv and just re-checks/updates.

set -euo pipefail
cd "$(dirname "$0")"

RUN_AFTER=1
for arg in "$@"; do
    case "$arg" in
        --no-run) RUN_AFTER=0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.11+ and re-run, or set PYTHON=/path/to/python." >&2
    exit 1
fi

# Python 3.11+ is required (the config loader uses the stdlib tomllib).
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python 3.11+ required (found $("$PYTHON" --version 2>&1)). Set PYTHON=/path/to/python3.11+." >&2
    exit 1
fi

echo "==> Using $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON")"

# Create the venv if it doesn't exist yet.
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment in .venv/"
    "$PYTHON" -m venv .venv
else
    echo "==> Reusing existing .venv/"
fi

VENV_PY=".venv/bin/python"

echo "==> Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

echo "==> Installing dependencies from requirements.txt"
"$VENV_PY" -m pip install -r requirements.txt

if [ "$RUN_AFTER" -eq 1 ]; then
    echo ""
    echo "==> Setup complete — launching the dashboard now."
    echo "    (opens in your browser and starts sweeping; press Ctrl-C to stop)"
    echo ""
    exec "$VENV_PY" main.py
else
    echo ""
    echo "==> Setup complete. Start the project with:"
    echo "      .venv/bin/python main.py"
fi
