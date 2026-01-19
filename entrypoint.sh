#!/bin/bash
set -e

# Logging function with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] $1"
}

log "=========================================="
log "Legal GenAI Application Starting"
log "=========================================="

# Marker file to track if initialization has been done
INIT_MARKER_FILE="${INIT_MARKER_PATH:-/app/data/.initialized}"
FORCE_REINIT="${FORCE_REINIT:-false}"

log "Configuration:"
log "  INIT_MARKER_PATH: $INIT_MARKER_FILE"
log "  FORCE_REINIT: $FORCE_REINIT"
log "  QDRANT_URL: ${QDRANT_URL:-not set}"
log "  COLLECTION_NAME: ${COLLECTION_NAME:-not set}"

# Create data directory if it doesn't exist
log "Ensuring data directory exists..."
mkdir -p "$(dirname "$INIT_MARKER_FILE")"

# Check if initialization is needed
if [ "$FORCE_REINIT" = "true" ]; then
    log "FORCE_REINIT is enabled. Running initialization..."
    python init.py
    
    # Create marker file with timestamp
    echo "Initialized on $(date -Iseconds)" > "$INIT_MARKER_FILE"
    log "Initialization complete. Marker file created at $INIT_MARKER_FILE"
elif [ ! -f "$INIT_MARKER_FILE" ]; then
    log "No marker file found at $INIT_MARKER_FILE. Running first-time initialization..."
    python init.py
    
    # Create marker file with timestamp
    echo "Initialized on $(date -Iseconds)" > "$INIT_MARKER_FILE"
    log "Initialization complete. Marker file created at $INIT_MARKER_FILE"
else
    log "Marker file exists at $INIT_MARKER_FILE. Skipping initialization."
    log "  Marker content: $(cat "$INIT_MARKER_FILE")"
fi

log "=========================================="
log "Starting Streamlit application..."
log "=========================================="
sleep 10

exec streamlit run app.py
