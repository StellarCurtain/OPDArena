#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--skip-flash-attn' ) ]]; then
  printf 'Usage: bash scripts/install.sh [--skip-flash-attn]\n' >&2
  exit 2
fi
python -c 'import platform, sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "The pinned runtime targets Linux x86_64"'
python -m pip install -r requirements.txt
if [[ "${1:-}" != '--skip-flash-attn' ]]; then
  python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
fi
python -m pip check
python -c 'import torch, pyarrow, transformers, vllm; print("Runtime imports OK; PyTorch", torch.__version__)'
