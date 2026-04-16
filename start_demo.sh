#!/bin/bash
# RungsX Causal Depth Demo — quick start
# Usage: ./start_demo.sh [TUNED_MODEL_URL]

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# Install deps if needed
pip3 install -q fastapi uvicorn openai pydantic requests 2>/dev/null || true

# Load API key
export $(grep -v '^#' "$DIR/.env" | xargs)

# Set tuned model URL if provided
if [ -n "$1" ]; then
  export TUNED_MODEL_URL="$1"
  echo "Using tuned model: $1"
else
  echo "No TUNED_MODEL_URL — right side will use GPT-4o-mini (fallback)"
fi

echo "Starting demo at http://localhost:9000"
cd "$DIR"
python3 causal_depth_demo.py
