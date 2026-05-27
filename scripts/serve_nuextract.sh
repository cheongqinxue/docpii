#!/usr/bin/env bash
# Serve NuExtract3 with vLLM (OpenAI-compatible API on port 8000).
#
# Requires: uv sync --extra vllm  (GPU + compatible CUDA driver)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-numind/NuExtract3}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES="${MAX_IMAGES:-16}"
ENABLE_MTP="${ENABLE_MTP:-false}"

args=(
  uv run vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}, \"video\": 0}"
  --chat-template-content-format openai
  --generation-config vllm
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$ENABLE_MTP" == "true" ]]; then
  args+=(--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}')
fi

exec "${args[@]}"
