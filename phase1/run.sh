#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace"
LOCAL_OUTPUT="$ROOT_DIR/output.json"
WORKSPACE_OUTPUT="$WORKSPACE_DIR/output.json"

echo "[run.sh] project root: $ROOT_DIR"
mkdir -p "$ROOT_DIR/benchmarks/generated" "$ROOT_DIR/logs"
cd "$ROOT_DIR"

if python3 src/agent.py; then
  :
else
  echo "[run.sh] agent returned non-zero status; checking whether output.json was still generated"
fi

if [[ -f "$LOCAL_OUTPUT" ]]; then
  if [[ -d "$WORKSPACE_DIR" && "$ROOT_DIR" != "$WORKSPACE_DIR" ]]; then
    cp "$LOCAL_OUTPUT" "$WORKSPACE_OUTPUT"
  fi
elif [[ -f "$WORKSPACE_OUTPUT" ]]; then
  :
else
  echo '{"target_spec_path":null,"total_metrics":0,"successful_analyses":0,"results":{},"agent_logs":["run.sh fallback: agent did not produce output.json"],"generated_benchmarks":[],"probe_plan":[],"normalized_targets":[],"triage":[],"ncu_analysis":[],"environment_notes":{"api_reported_device_properties":{},"observed_active_sms":null,"frequency_lock_suspected":null,"sm_masking_suspected":null}}' > "$LOCAL_OUTPUT"
  if [[ -d "$WORKSPACE_DIR" && "$ROOT_DIR" != "$WORKSPACE_DIR" ]]; then
    cp "$LOCAL_OUTPUT" "$WORKSPACE_OUTPUT"
  fi
fi

if [[ -f "$LOCAL_OUTPUT" || -f "$WORKSPACE_OUTPUT" ]]; then
  echo "[run.sh] output.json generated"
  exit 0
fi

echo "[run.sh] failed to generate output.json"
exit 1
