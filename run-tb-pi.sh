#!/bin/bash
# Run Terminal-Bench 2.0 with pi agent + local llama.cpp
# Usage: ./run-tb-pi.sh [task-filter] [num-tasks]

set -euo pipefail

# --- Configuration ---
# Podman socket (user-level, avoids docker.sock permission issues)
export DOCKER_HOST="unix:///run/user/1000/podman/podman.sock"

# Local llama.cpp server
export OPENAI_BASE_URL="http://host.containers.internal:8081/v1"
export OPENAI_API_KEY="llama-cpp-local"  # dummy key, pi requires it for openai provider

# Model (must be in provider/name format)
MODEL="local-llama/Qwen3.6-35B-A3B-spec2"

# Agent
AGENT="pi"

# Dataset
DATASET="terminal-bench/terminal-bench-2"

# --- Arguments ---
TASK_FILTER="${1:-}"          # e.g., "terminal-bench/fix-git" or glob pattern
NUM_TASKS="${2:-1}"           # number of tasks to run

# --- Build task filter ---
if [ -n "$TASK_FILTER" ]; then
    TASK_FILTER_FLAG="-i $TASK_FILTER"
else
    TASK_FILTER_FLAG=""
fi

# --- Run ---
echo "============================================"
echo "  Terminal-Bench 2.0 + pi + llama.cpp"
echo "============================================"
echo "  Model:        $MODEL"
echo "  Base URL:     $OPENAI_BASE_URL"
echo "  Agent:        $AGENT"
echo "  Dataset:      $DATASET"
echo "  Task filter:  ${TASK_FILTER:-all}"
echo "  Num tasks:    $NUM_TASKS"
echo "  Docker host:  $DOCKER_HOST"
echo "============================================"
echo ""

harbor run \
    -d "$DATASET" \
    -a "$AGENT" \
    -m "$MODEL" \
    -l "$NUM_TASKS" \
    $TASK_FILTER_FLAG \
    --ae "OPENAI_BASE_URL=$OPENAI_BASE_URL" \
    --ae "OPENAI_API_KEY=$OPENAI_API_KEY"
