#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import torch
PY
then
  if [ -x "${HOME}/miniconda3/bin/python3" ]; then
    export PATH="${HOME}/miniconda3/bin:${PATH}"
  fi
fi

export ENABLE_LLM="${ENABLE_LLM:-1}"
export ENABLE_NCU="${ENABLE_NCU:-1}"
export FULL_SEARCH="${FULL_SEARCH:-1}"
export RANDOM_SIZES="${RANDOM_SIZES:-0}"
export AGENT_TIME_BUDGET_SECONDS="${AGENT_TIME_BUDGET_SECONDS:-1500}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-30}"
export LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-200}"
export NCU_PROFILE="${NCU_PROFILE:-1}"
export NCU_PROFILE_D="${NCU_PROFILE_D:-4096}"
export NCU_TIMEOUT_S="${NCU_TIMEOUT_S:-180}"
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="${PWD}/torch_extensions_cache"

echo "[run.sh] pwd=$(pwd)"
echo "[run.sh] python=$(python3 --version 2>&1 || true)"
echo "[run.sh] ENABLE_LLM=${ENABLE_LLM}"
echo "[run.sh] ENABLE_NCU=${ENABLE_NCU}"
echo "[run.sh] FULL_SEARCH=${FULL_SEARCH}"
echo "[run.sh] RANDOM_SIZES=${RANDOM_SIZES}"
echo "[run.sh] AGENT_TIME_BUDGET_SECONDS=${AGENT_TIME_BUDGET_SECONDS}"
echo "[run.sh] LLM_TIMEOUT_S=${LLM_TIMEOUT_S}"
echo "[run.sh] NCU_PROFILE=${NCU_PROFILE}"
echo "[run.sh] NCU_PROFILE_D=${NCU_PROFILE_D}"
echo "[run.sh] NCU_TIMEOUT_S=${NCU_TIMEOUT_S}"
echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

python3 agent.py "$@"
status=$?

if [ ! -f optimized_lora.cu ]; then
  echo "[run.sh] optimized_lora.cu missing; emitting baseline fallback"
  python3 agent.py --emit-baseline-only || true
fi

if [ -f output.json ]; then
  python3 - <<'PY' || true
import json
from pathlib import Path

data = json.loads(Path("output.json").read_text(encoding="utf-8"))
summary = data.get("summary", {})
print(f"[run.sh] best={summary.get('best_candidate')} speedup={summary.get('best_speedup')} correct={summary.get('correct')}")
print(f"[run.sh] candidates={summary.get('num_candidates_tested')} compile_ok={summary.get('num_compile_ok')} correct_count={summary.get('num_correct')}")
print(f"[run.sh] llm={data.get('llm')}")
print(f"[run.sh] ncu={data.get('ncu')}")
print(f"[run.sh] output_json_bytes={Path('output.json').stat().st_size}")
PY
fi

if [ -f optimized_lora.cu ]; then
  exit 0
fi
exit "${status}"
