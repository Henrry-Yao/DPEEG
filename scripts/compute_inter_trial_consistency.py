#!/usr/bin/env python3
"""
计算每位受试者真实EEG试次间一致性（inter-trial cosine similarity）。

对于每张测试图像，取 80 次重复试次的 EEG 信号，计算所有试次对的余弦相似度均值。
最终汇总为每位受试者 200 张图像的平均试次间一致性，以及与信号方差的对照。
"""
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("data/things-eeg/Preprocessed_data_250Hz_whiten")
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
TIMESTEPS = (0, 250)

# 选用的通道（与 config 一致）
SELECTED_CH = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
               'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2']


def compute_subject_consistency(subject: str) -> dict:
    """返回 {subject, mean_consistency, std_consistency, per_image_consistencies, eeg_var}"""
    data_path = DATA_DIR / subject / "test.pt"
    loaded = torch.load(data_path, weights_only=False)
    eeg_raw = torch.from_numpy(loaded['eeg']).float()  # (N_img, N_trials, C, T)

    # 选通道 + 截时间
    selected_idx = [ALL_CHANNELS.index(ch) for ch in SELECTED_CH]
    eeg_raw = eeg_raw[:, :, selected_idx, TIMESTEPS[0]:TIMESTEPS[1]]  # (200, 80, 17, 250)

    n_images, n_trials = eeg_raw.shape[0], eeg_raw.shape[1]

    # 信号方差（与论文一致）
    eeg_var = eeg_raw.var(dim=1).mean(dim=(-1, -2)).numpy().mean()  # scalar

    # 逐图像计算试次间一致性
    per_image_consistencies = []
    for i in range(n_images):
        trials = eeg_raw[i]  # (80, 17, 250)
        trials_flat = trials.reshape(n_trials, -1)  # (80, 17*250)
        # 余弦相似度矩阵
        normed = trials_flat / (trials_flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = normed @ normed.T  # (80, 80)
        # 只取上三角的非对角线元素
        triu_idx = torch.triu_indices(n_trials, n_trials, offset=1)
        pairwise_sims = sim_matrix[triu_idx[0], triu_idx[1]]
        per_image_consistencies.append(pairwise_sims.mean().item())

    per_image = np.array(per_image_consistencies)
    return {
        "subject": subject,
        "mean_consistency": float(per_image.mean()),
        "std_consistency": float(per_image.std()),
        "per_image": per_image,
        "eeg_var": float(eeg_var),
    }


def main():
    results = []
    for sub in SUBJECTS:
        print(f"Processing {sub}...", end=" ", flush=True)
        r = compute_subject_consistency(sub)
        results.append(r)
        print(f"consistency={r['mean_consistency']:.4f} ± {r['std_consistency']:.4f}, var={r['eeg_var']:.4f}")

    # ── 汇总输出 ──
    print("\n" + "=" * 85)
    print(f"{'Subject':<10} {'Inter-trial cos':>18} {'Std':>10} {'EEG Var':>12} {'Notes'}")
    print("-" * 85)
    for r in results:
        notes = ""
        if r["subject"] == "sub-09":
            notes = "  ← 高方差 + 高一致性？"
        elif r["subject"] == "sub-05":
            notes = "  ← 高方差"
        print(f"{r['subject']:<10} {r['mean_consistency']:>15.4f} ± {r['std_consistency']:>7.4f}  {r['eeg_var']:>10.4f}{notes}")
    print("=" * 85)

    # 对齐论文的方差分组
    variances = [r["eeg_var"] for r in results]
    median_var = np.median(variances)
    print(f"\nMedian EEG variance: {median_var:.4f}")
    for r in results:
        level = "High" if r["eeg_var"] > median_var * 1.1 else ("Low" if r["eeg_var"] < median_var * 0.9 else "Medium")
        print(f"  {r['subject']}: var={r['eeg_var']:.4f} → {level}")

    # sub-09 vs 其他人对比
    sub09 = [r for r in results if r["subject"] == "sub-09"][0]
    others_mean = np.mean([r["mean_consistency"] for r in results if r["subject"] != "sub-09"])
    others_eeg_var = np.mean([r["eeg_var"] for r in results if r["subject"] != "sub-09"])
    print(f"\nsub-09 试次间一致性: {sub09['mean_consistency']:.4f}  vs  其他受试者均值: {others_mean:.4f}")
    print(f"sub-09 EEG方差:      {sub09['eeg_var']:.4f}  vs  其他受试者均值: {others_eeg_var:.4f}")


if __name__ == "__main__":
    main()
