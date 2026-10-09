#!/bin/bash
# DPEEG — training script for THINGS-EEG (intra-subject setting).
#
# The three task directions from the paper (Sec. 4):
#   1) Decoding / retrieval-based encoding — contrastive alignment only (50 epochs)
#   2) Generative encoding                   — two-stage decoupled training (230 epochs)
#   3) Joint training (ablation)             — contrastive + diffusion optimized together
#
# Usage:
#   bash scripts/exp.sh                 # decoding + generative encoding, all 10 subjects
#   bash scripts/exp.sh --mode decode   # decoding / retrieval-based encoding only
#   bash scripts/exp.sh --mode gen      # generative encoding only
#   bash scripts/exp.sh --mode joint    # joint training (ablation) only

set -e
cd "$(dirname "$0")/.."

brain_backbone="EEGProjectLayer"
vision_backbone="RN50"
dataset="eeg"
seed=0
config="configs/eeg/ubp.yaml"

# Parse --mode <name>
mode="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

subjects=""
for i in $(seq -w 1 10); do
  subjects="${subjects} sub-${i}"
done

run_decode() {
  local sub="$1"
  echo "=== ${sub} decoding / retrieval-based encoding ==="
  python main.py \
    --config "$config" \
    --dataset "$dataset" \
    --subjects "$sub" \
    --seed "$seed" \
    --exp_setting intra-subject \
    --brain_backbone "$brain_backbone" \
    --vision_backbone "$vision_backbone" \
    --epoch 50 \
    --lr 1e-4
}

run_gen() {
  local sub="$1"
  echo "=== ${sub} generative encoding (two-stage decoupled) ==="
  python main.py \
    --config "$config" \
    --dataset "$dataset" \
    --subjects "$sub" \
    --seed "$seed" \
    --exp_setting intra-subject \
    --brain_backbone "$brain_backbone" \
    --vision_backbone "$vision_backbone" \
    --epoch 230 \
    --lr 1e-4 \
    --generative_encoding
}

run_joint() {
  local sub="$1"
  echo "=== ${sub} joint training (ablation) ==="
  python main.py \
    --config "$config" \
    --dataset "$dataset" \
    --subjects "$sub" \
    --seed "$seed" \
    --exp_setting intra-subject \
    --brain_backbone "$brain_backbone" \
    --vision_backbone "$vision_backbone" \
    --epoch 230 \
    --lr 1e-4 \
    --generative_encoding \
    --joint_training
}

for sub in $subjects; do
  case "$mode" in
    decode) run_decode "$sub" ;;
    gen)    run_gen "$sub" ;;
    joint)  run_joint "$sub" ;;
    all)    run_decode "$sub"; run_gen "$sub" ;;
    *)      echo "Unknown mode: $mode (expected decode|gen|joint|all)"; exit 1 ;;
  esac
done

echo "DONE"
