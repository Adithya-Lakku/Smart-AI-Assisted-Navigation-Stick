#!/usr/bin/env bash
# ==============================================================================
# Smart AI Navigation Stick - Local Inference Server Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "[Server Launcher] Loading environment from .env..."
    export $(grep -v '^#' .env | xargs)
fi

echo "=================================================================="
echo " Starting Smart AI Navigation Stick (Local AI Server)"
echo "=================================================================="
echo " Working Directory : $PROJECT_ROOT"
echo " Python Executable : $(which python3)"
echo " Inference Engine  : ${STICK_INFERENCE_ENGINE:-local}"
echo " TTS Engine        : ${STICK_TTS_ENGINE:-piper}"
echo " Port              : ${STICK_SERVER_PORT:-5000}"
echo "=================================================================="

# Check if llama-server is needed
if [ "${STICK_INFERENCE_ENGINE}" == "local" ]; then
    echo "[Info] Configured for local llama.cpp inference."
    if ! curl -s --connect-timeout 2 http://localhost:8080/health >/dev/null 2>&1; then
        echo "[Notice] Local llama-server not detected on port 8080."
        echo "         If running standalone llama-server, launch it with:"
        echo "         llama-server -m models/gemma-4b-it-q4_k_m.gguf --mmproj models/gemma-mmproj-f16.gguf --port 8080"
        echo "         (Server will use heuristic fallback or mock mode if unavailable)"
    fi
fi

exec python3 -m server.app "$@"
