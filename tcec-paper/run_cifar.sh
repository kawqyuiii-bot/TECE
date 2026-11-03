#!/usr/bin/env bash
# Paper Table 9. usage: ./run_cifar.sh <a_bits: 8|6> <steps: 100|250> [official_ckpt.pth]
# The quantized model is calibrated once, at DDIM-100, then reused for 250 steps.
# env: N = images per mode (default 50000), BS = batch size (default 256)
set -euo pipefail
cd "$(dirname "$0")"
AB=${1:-8}; STEPS=${2:-100}; CKPT=${3:-}; N=${N:-50000}; BS=${BS:-256}
ROOT=third_party/q-diffusion; O=runs/cifar/w4a$AB; mkdir -p "$O"
export PYTHONPATH="$PWD/$ROOT:${PYTHONPATH:-}"
COMMON=(--w_bits 4 --a_bits "$AB" --qdiff_root "$ROOT" --steps "$STEPS")

[ -f runs/cifar/cali.pt ] || python cifar_tcec.py gen-cali --steps 100 --n 256 --out runs/cifar/cali.pt

if [ -z "$CKPT" ]; then
  CKPT="$O/ckpt.pth"
  if [ ! -f "$CKPT" ]; then
    ( cd "$ROOT" && python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained \
        --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff \
        --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit "$AB" --a_sym --split \
        --cali_data_path "$OLDPWD/runs/cifar/cali.pt" -l "$OLDPWD/$O/qdiff" --max_images 0 )
    cp "$(ls -t "$O"/qdiff/samples/*/ckpt.pth | head -1)" "$CKPT"
  fi
fi

K="$O/K_s$STEPS.pt"
[ -f "$K" ] || python cifar_tcec.py calibrate-k "${COMMON[@]}" --cali_ckpt "$CKPT" --n 1024 --seed 777777 --out "$K"

for MODE in none paper; do
  D="$O/s$STEPS/$MODE"
  python cifar_tcec.py sample "${COMMON[@]}" --cali_ckpt "$CKPT" --k "$K" --mode "$MODE" --n "$N" --bs "$BS" --outdir "$D"
  fidelity --gpu 0 --isc --fid --input1 "$D" --input2 cifar10-train | tee "$D.metrics.txt"
  python eval_adm.py --samples "$D" --out "$D.adm.txt"             # IS, FID, sFID
done
