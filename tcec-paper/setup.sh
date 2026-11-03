#!/usr/bin/env bash
# Pinned to the commits that were read when this code was written.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party
get() { [ -d "third_party/$1" ] || { git clone "https://github.com/$2.git" "third_party/$1"; git -C "third_party/$1" checkout "$3"; }; }
get q-diffusion Xiuyu-Li/q-diffusion 715783d
get guided-diffusion openai/guided-diffusion 22e0df8
# constraints.txt pins torch 2.5 and the libraries that must match it
pip install -c constraints.txt torch torchvision einops omegaconf pytorch-lightning torch-fidelity pyyaml diffusers transformers huggingface_hub accelerate torchmetrics
[ -d .venv-adm ] || python -m venv .venv-adm                     # sFID evaluator, TensorFlow on CPU
.venv-adm/bin/pip install tensorflow-cpu "scipy<1.14" requests tqdm
if [ "${1:-}" = "sdxl" ]; then
  get deepcompressor mit-han-lab/deepcompressor 69f3473
  pip install -c constraints.txt -e third_party/deepcompressor
  cp configs/sdxl.yaml third_party/deepcompressor/examples/diffusion/configs/model/sdxl.yaml
  cp configs/int8.yaml third_party/deepcompressor/examples/diffusion/configs/svdquant/int8.yaml
fi
