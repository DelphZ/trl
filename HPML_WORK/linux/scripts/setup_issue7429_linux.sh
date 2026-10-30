#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python3 -m venv .venv-issue7429
source .venv-issue7429/bin/activate
python -m pip install --upgrade pip
python -m pip install -r HPML_WORK/linux/requirements.txt

python -c 'import torch; from trl.import_utils import is_vllm_available; assert torch.cuda.is_available(), "CUDA GPU is not visible to PyTorch"; assert is_vllm_available(), "vLLM failed to import"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__); import vllm; print("vLLM:", vllm.__version__)'
python HPML_WORK/shared/scripts/download_qwen3.py --local-dir models/Qwen3-0.6B
