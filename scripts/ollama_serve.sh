#!/bin/bash
# Start an Ollama server inside the existing apptainer image (no rebuild) and
# advertise its IP:port to a file the analysis code can read.
set -u
SIF="/home/mickaelz/apptainer-ollama/ollama.sif"
ENDPOINT_FILE="/home/mickaelz/Network analysis/logs/ollama_endpoint.txt"

PORT=$(shuf -i 9980-9999 -n 1)
IP=$(hostname -i | awk '{print $1}')
echo "OLLAMA ENDPOINT: ${IP}:${PORT}"
mkdir -p "$(dirname "$ENDPOINT_FILE")"
echo "${IP}:${PORT}" > "$ENDPOINT_FILE"

export OLLAMA_HOST="${IP}:${PORT}"
# Serve (this call blocks for the lifetime of the job, keeping the API up).
apptainer exec --nv --writable-tmpfs "$SIF" bash -c "export OLLAMA_HOST=${IP}:${PORT} && ollama serve"
