#!/usr/bin/env python3
"""Multi-seed MLP baseline training (seeds 0, 1, 2) for robustness check."""
import subprocess, sys, os, json
from pathlib import Path
import numpy as np

PY = sys.executable
PROJ = str(Path(__file__).resolve().parents[1])
SCR = os.path.join(PROJ, "scripts", "train_mlp_baseline.py")
CKPT_DIR = os.path.join(PROJ, "exp", "eeg_intra-subject_ubp_EEGProjectLayer_RN50")

subjects = ["sub-%02d" % i for i in range(1, 11)]
losses = ["mse", "cosine"]
seeds = [0, 1, 2]

for sub in subjects:
    ckpt = os.path.join(CKPT_DIR, "%s_seed0_genFalse/checkpoints/last.ckpt" % sub)
    for loss in losses:
        for seed in seeds:
            out_dir = os.path.join(PROJ, "exp/mlp_baseline", "%s_%s_RN50_s%d" % (sub, loss, seed))
            if os.path.exists(os.path.join(out_dir, "results.json")):
                print("[SKIP] %s %s seed=%d" % (sub, loss.upper(), seed))
                continue
            print("[RUN] %s %s seed=%d" % (sub, loss.upper(), seed), flush=True)
            cmd = [PY, SCR, "--subjects", sub, "--loss", loss,
                   "--brain_ckpt", ckpt, "--seed", str(seed)]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=PROJ, timeout=600)
            if r.returncode != 0:
                print("  ERROR: %s" % r.stderr[-200:])

# ── Summary ──
print("\n" + "=" * 85)
for loss in ["mse", "cosine"]:
    print("MLP-%s multi-seed summary (3 seeds):" % loss.upper())
    print("%-10s %8s %8s %8s %10s" % ("Subject", "Seed0", "Seed1", "Seed2", "Mean±Std"))
    print("-" * 60)
    all_vals = []
    for sub in subjects:
        vals = []
        for seed in seeds:
            fpath = os.path.join(PROJ, "exp/mlp_baseline", "%s_%s_RN50_s%d" % (sub, loss, seed), "results.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    vals.append(json.load(f)["top1"])
            else:
                vals.append(None)
        if all(v is not None for v in vals):
            all_vals.extend(vals)
            print("%-10s %7.1f%% %7.1f%% %7.1f%%  %5.1f%%±%.1f" % (
                sub, vals[0], vals[1], vals[2], np.mean(vals), np.std(vals, ddof=1)))
        else:
            print("%-10s (incomplete)" % sub)
    if all_vals:
        print("-" * 60)
        print("%-10s %43s  %5.1f%%±%.1f (n=%d)" % ("Overall", "", np.mean(all_vals), np.std(all_vals, ddof=1), len(all_vals)))
    print()
print("DONE")
