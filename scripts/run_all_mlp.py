#!/usr/bin/env python3
"""Batch run MLP-MSE and MLP-COS for all 10 subjects."""
import subprocess, sys, os, json

PY = "/home/henrry/miniconda3/envs/ubp/bin/python"
PROJ = "/home/henrry/UBP/Uncertainty-aware-Blur-Prior-main"
SCR = os.path.join(PROJ, "scripts/train_mlp_baseline.py")
CKPT_DIR = os.path.join(PROJ, "exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50")

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = {"mse": {}, "cosine": {}}

for sub in subjects:
    ckpt = os.path.join(CKPT_DIR, f"{sub}_seed0_genFalse/checkpoints/last.ckpt")
    for loss in ["mse", "cosine"]:
        out_dir = os.path.join(PROJ, "exp/mlp_baseline", f"{sub}_{loss}_RN50")
        if os.path.exists(os.path.join(out_dir, "results.json")):
            print(f"[SKIP] {sub} {loss} — already done")
            continue
        print(f"[RUN] {sub} {loss} ...", flush=True)
        cmd = [PY, SCR, "--subjects", sub, "--loss", loss,
               "--brain_ckpt", ckpt, "--seed", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJ, timeout=600)
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr[-300:]}")
        else:
            try:
                with open(os.path.join(out_dir, "results.json")) as f:
                    d = json.load(f)
                print(f"  Top1={d['top1']}% diag={d['diag']} off={d['off']} ratio={d['diag_off_ratio']}")
            except:
                print(f"  FINISHED but no results.json")

# ── Final summary ──
print("\n" + "=" * 85)
for loss in ["MSE", "COSINE"]:
    loss_key = loss.lower()
    print(f"{'Subject':<10} {'MLP-'+loss:>15}")
    print("-" * 30)
    tops = []
    for i in range(1, 11):
        sub = f"sub-{i:02d}"
        fpath = os.path.join(PROJ, "exp/mlp_baseline", f"{sub}_{loss_key}_RN50", "results.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            tops.append(d['top1'])
            print(f"{sub:<10} {d['top1']:>10.1f}%")
        else:
            print(f"{sub:<10} {'MISSING':>10}")
    if tops:
        print(f"{'Avg':<10} {sum(tops)/len(tops):>10.1f}%")
    print()
print("DONE")
