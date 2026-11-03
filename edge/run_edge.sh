#!/usr/bin/env bash
# ==============================================================================
# Smart AI Navigation Stick - Raspberry Pi Edge Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "[Edge Launcher] Loading environment from .env..."
    export $(grep -v '^#' .env | xargs)
fi

echo "=================================================================="
echo " Starting Smart AI Navigation Stick (Edge Client)"
echo "=================================================================="
echo " Working Directory : $PROJECT_ROOT"
echo " Python Executable : $(which python3)"
echo " Server URL        : ${STICK_EDGE_SERVER_URL:-http://192.168.1.100:5000}"
echo " Simulate Hardware : ${STICK_SIMULATE_HARDWARE:-false}"
echo "=================================================================="

exec python3 -m edge.main "$@"
