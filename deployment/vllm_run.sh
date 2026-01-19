#!/bin/bash

set -e  # Exit on any error

# Ensure script is run with bash (not sh)
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set TMPDIR to a location with more space (use script directory instead of /tmp)
export TMPDIR="$SCRIPT_DIR/.tmp"
mkdir -p "$TMPDIR"
echo "Using temporary directory: $TMPDIR"

# Clean up any existing temp files
if [ -d "$TMPDIR" ]; then
    echo "Cleaning up temporary files..."
    rm -rf "$TMPDIR"/*
fi

# Load environment variables from .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

echo "=== vLLM Installation and Server Setup ==="

# Check if Python is available (try python3 first, then python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Neither python3 nor python is installed. Please install Python 3.10-3.13 first."
    exit 1
fi

# Check Python version (vLLM requires 3.10-3.13)
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION (using $PYTHON_CMD)"

# Check if uv is available (recommended method)
if command -v uv &> /dev/null; then
    uv venv --python 3.12 --seed
    echo "Waiting 5 seconds..."
    sleep 5
    source .venv/bin/activate

    echo "Using uv to install vLLM (recommended method)..."
    uv pip install vllm --torch-backend=auto
else
    echo "Error: uv is not available. Please install uv first."
    exit 1
fi

# Verify vLLM installation
if ! $PYTHON_CMD -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM installation failed. Please check the error messages above."
    exit 1
fi

echo "vLLM installed successfully!"

uv pip install timm

# Clean up temporary files after installation
echo "Cleaning up temporary files..."
rm -rf "$TMPDIR"/*


# Check if CUDA/GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

echo ""
echo "=== Starting vLLM Server ==="
echo "Model: nvidia/Llama-3.1-8B-Instruct-FP8"
echo "Host: 0.0.0.0"
echo "Port: 8088"
echo "GPU Memory Utilization: 0.85 (85%)"
echo "Max Model Length: 3072"
echo "Max Num Seqs: 5"
echo "Prefix Caching: Disabled"
echo ""
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8\
    --host 0.0.0.0 \
    --port 8088 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 10 \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0}' \
    --kv-cache-dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template ./tool_chat_template_llama3.1_json.jinja

 