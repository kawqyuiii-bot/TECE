#!/usr/bin/env bash
# Paper Table 1. usage: ./run_sdxl.sh <4|8> (needs ./setup.sh sdxl first)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B=${1:-4}; NAME=sdxl-w${B}a${B}
QCFG=configs/svdquant/int$B.yaml
cd "$HERE/third_party/deepcompressor/examples/diffusion"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
M=configs/model/sdxl.yaml
MODEL="$HERE/runs/sdxl/$NAME/model"; K="$HERE/runs/sdxl/$NAME/K.pt"; mkdir -p "$MODEL"
RUN="python $HERE/sdxl_tcec.py"

$RUN ptq $M --output-dirname reference                 # FP reference images for PSNR
$RUN collect $M configs/collect/qdiff.yaml             # SVDQuant calibration data, 128 prompts
$RUN ptq $M "$QCFG" --save-model "$MODEL" --output-dirname "$NAME-svdquant"
TCEC_K="$K" $RUN calibrate-k $M "$QCFG" --load-from "$MODEL" --output-dirname "$NAME-k"
TCEC_K="$K" TCEC_MODE=paper $RUN ptq $M "$QCFG" --load-from "$MODEL" --output-dirname "$NAME-tcec"
