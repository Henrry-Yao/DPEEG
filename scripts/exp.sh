#!/bin/bash

brain_backbone="EEGProjectLayer"
vision_backbone="RN50"
dataset="eeg"
seed=0

for i in {01..10}; do
    echo "=== Running sub-$i ==="
    python main.py \
        --config configs/eeg/ubp.yaml \
        --dataset $dataset \
        --subjects sub-$i \
        --seed $seed \
        --exp_setting intra-subject \
        --brain_backbone $brain_backbone \
        --vision_backbone $vision_backbone \
        --epoch 50 \
        --lr 1e-4
done