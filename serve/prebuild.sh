#!/usr/bin/env bash
# Build the CUDA extensions on this node, in the serving container: prebuild.sh  (ML4_TREE as in cp4_serve.sh).
set -eu
R="${ML4_TREE:?set ML4_TREE}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$R/logs"
cp "$HERE/prebuild.py" "$R/logs/prebuild.py"
docker run --rm --name ml4-prebuild --gpus all --network none --ipc host \
  -e TORCH_EXTENSIONS_DIR=/w/torch-ext -e TRITON_CACHE_DIR=/w/triton -v "$R:/w" -w /w \
  nvcr.io/nvidia/pytorch:26.07-py3 python3 -u /w/logs/prebuild.py
