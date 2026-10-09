#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=python
SCR=scripts/train_mlp_baseline.py
CKPT_DIR=exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50

for sub in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-10; do
    CKPT="${CKPT_DIR}/${sub}_seed0_genFalse/checkpoints/last.ckpt"
    echo "=== ${sub} MLP-MSE ==="
    ${PY} ${SCR} --subjects ${sub} --loss mse --brain_ckpt ${CKPT} --seed 0
    echo "=== ${sub} MLP-COS ==="
    ${PY} ${SCR} --subjects ${sub} --loss cosine --brain_ckpt ${CKPT} --seed 0
done
echo "ALL DONE"
