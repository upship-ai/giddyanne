#!/bin/bash
# Benchmark server startup time from cold start (cleared index) to ready
#
# This script:
# 1. Stops any running server
# 2. Clears the index
# 3. Starts the server
# 4. Polls /status until state="ready"
# 5. Reports elapsed time

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8000
POLL_INTERVAL=0.1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[benchmark]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[benchmark]${NC} $1"
}

error() {
    echo -e "${RED}[benchmark]${NC} $1" >&2
}

# Stop any running server
if giddy status &>/dev/null; then
    log "Stopping existing server..."
    giddy down
    sleep 0.5
fi

# Clear the index
log "Clearing index..."
rm -rf .giddyanne/vectors.lance

# Record start time (nanoseconds for precision)
START_TIME=$(python3 -c 'import time; print(time.perf_counter())')

# Start server in verbose mode (foreground, logs to stderr)
log "Starting server..."
.venv/bin/python main.py 2>&1 &
SERVER_PID=$!

# Cleanup on exit
cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        log "Stopping server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for /health to respond (server is up)
log "Waiting for server to be available..."
while ! curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        error "Server process died!"
        exit 1
    fi
    sleep $POLL_INTERVAL
done
HEALTH_TIME=$(python3 -c 'import time; print(time.perf_counter())')

log "Server responding, waiting for indexing to complete..."

# Poll /status until state="ready"
while true; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        error "Server process died!"
        exit 1
    fi

    STATUS=$(curl -s "http://127.0.0.1:$PORT/status" 2>/dev/null || echo '{}')
    STATE=$(echo "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("state",""))')

    if [ "$STATE" = "ready" ]; then
        break
    fi

    # Show progress if indexing
    if [ "$STATE" = "indexing" ]; then
        INDEXED=$(echo "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("indexed",0))')
        TOTAL=$(echo "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("total",0))')
        PERCENT=$(echo "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("percent",0))')
        printf "\r${GREEN}[benchmark]${NC} Indexing: %d/%d (%.0f%%)   " "$INDEXED" "$TOTAL" "$PERCENT"
    fi

    sleep $POLL_INTERVAL
done

# Clear the progress line
printf "\r%80s\r" ""

END_TIME=$(python3 -c 'import time; print(time.perf_counter())')

# Calculate elapsed times
TOTAL_ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
HEALTH_ELAPSED=$(python3 -c "print(f'{$HEALTH_TIME - $START_TIME:.2f}')")
INDEX_ELAPSED=$(python3 -c "print(f'{$END_TIME - $HEALTH_TIME:.2f}')")

echo ""
echo "=========================================="
echo "         Startup Benchmark Results        "
echo "=========================================="
echo ""
echo "  Server available:  ${HEALTH_ELAPSED}s"
echo "  Indexing complete: ${INDEX_ELAPSED}s"
echo "  ─────────────────────────────"
echo "  Total time:        ${TOTAL_ELAPSED}s"
echo ""
echo "=========================================="

# Keep server running or stop it?
log "Benchmark complete. Stopping server."
