#!/usr/bin/env python3
"""
PCA: Real EEG / Diffusion Gen / MLP-Cos Gen / Image CLIP — all in one space.
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base.data_eeg import EEGDataset
from base.diffusion import DiffusionModel
from base.utils import instantiate_from_config
from scripts.train_mlp_baseline import MLPGenerator
from omegaconf import OmegaConf

PRETRAIN_Z = {"RN50": 1024, "RN101": 512, "ViT-B-16": 512, "ViT-B-32": 512,
              "ViT-L-14": 768, "ViT-H-14": 1024, "ViT-g-14": 1024, "ViT-bigG-14": 1280}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def strip_prefix(d, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in d.items() if k.startswith(p)}


def build_config(subject, vision_backbone="RN50"):
    config = OmegaConf.load("configs/eeg/ubp.yaml")
    config["dataset"] = "eeg"
    config["subjects"] = subject
    config["exp_setting"] = "intra-subject"
    config["brain_backbone"] = "EEGProjectLayer"
    config["vision_backbone"] = vision_backbone
    config["z_dim"] = PRETRAIN_Z[vision_backbone]
    config["c"] = 6
    config["data"]["subjects"] = [subject]
    config["data"]["model_type"] = vision_backbone
    config["data"]["test_batch_size"] = 200
    config["generative_encoding"] = True
    return config


@torch.no_grad()
def collect_all_features(config, brain, diffusion, mlp_cos, device, max_samples=200):
    dataset = EEGDataset(config, mode="test")
    loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=0)

    real_eeg, gen_diff, gen_mlp, img_clip = [], [], [], []

    for batch in loader:
        eeg = batch["eeg"].to(device).float()
        img_z = batch["img_features"].to(device).float()
        img_z = F.normalize(img_z, dim=-1)

        eeg_z = brain(eeg)
        eeg_z = F.normalize(eeg_z, dim=-1)

        diff_z = diffusion(img_z)
        diff_z = F.normalize(diff_z, dim=-1)

        mlp_z = mlp_cos(img_z)
        mlp_z = F.normalize(mlp_z, dim=-1)

        real_eeg.append(eeg_z.cpu())
        gen_diff.append(diff_z.cpu())
        gen_mlp.append(mlp_z.cpu())
        img_clip.append(img_z.cpu())

        if sum(t.shape[0] for t in real_eeg) >= max_samples:
            break

    return {
        "Real EEG":         torch.cat(real_eeg, dim=0)[:max_samples].numpy(),
        "Diffusion Gen":    torch.cat(gen_diff, dim=0)[:max_samples].numpy(),
        "MLP-Cos Gen":      torch.cat(gen_mlp, dim=0)[:max_samples].numpy(),
        "Image CLIP":       torch.cat(img_clip, dim=0)[:max_samples].numpy(),
    }


def pca_project(features):
    labels = list(features.keys())
    sizes = [features[k].shape[0] for k in labels]
    x = np.concatenate([features[k] for k in labels], axis=0)
    x_c = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_c, full_matrices=False)
    x_2d = x_c @ vt[:2].T
    out = {}
    start = 0
    for label, size in zip(labels, sizes):
        out[label] = x_2d[start:start + size]
        start += size
    return out


def compute_summary(features):
    real = features["Real EEG"]
    diff = features["Diffusion Gen"]
    mlp  = features["MLP-Cos Gen"]
    def pair_stats(gen, ref):
        sim = gen @ ref.T
        d = float(np.diag(sim).mean())
        off_mask = ~np.eye(sim.shape[0], dtype=bool)
        o = float(sim[off_mask].mean())
        return d, o

    d_diff_real, o_diff_real = pair_stats(diff, real)
    d_mlp_real, o_mlp_real = pair_stats(mlp, real)
    return {
        "diff_diag": d_diff_real, "diff_off": o_diff_real, "diff_ratio": d_diff_real/o_diff_real,
        "mlp_diag": d_mlp_real,   "mlp_off": o_mlp_real,   "mlp_ratio": d_mlp_real/o_mlp_real,
    }


def plot_pca(projected, features, subject, out_path, summary):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    styles = {
        "Image CLIP":      dict(color="#377eb8", marker="x", s=50, alpha=0.70, label="Image CLIP feature"),
        "Real EEG":        dict(color="#2ca25f", marker="o", s=42, alpha=0.70, label="Real EEG feature"),
        "Diffusion Gen":   dict(color="#8e44ad", marker="^", s=48, alpha=0.82, label="Diffusion generated EEG"),
        "MLP-Cos Gen":     dict(color="#e41a1c", marker="D", s=48, alpha=0.82, label="MLP-Cosine generated EEG"),
    }

    for key in ["Image CLIP", "Real EEG", "Diffusion Gen", "MLP-Cos Gen"]:
        pts = projected[key]
        ax.scatter(pts[:, 0], pts[:, 1], edgecolors="none", **styles[key])

    ax.set_title(f"Feature-space Distribution ({subject}, PCA)\n"
                 f"MLP-Cos: diag={summary['mlp_diag']:.3f} off={summary['mlp_off']:.3f} ratio={summary['mlp_ratio']:.1f} | "
                 f"Diff: diag={summary['diff_diag']:.3f} off={summary['diff_off']:.3f} ratio={summary['diff_ratio']:.1f}",
                 weight="bold", fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(color="#d9d9d9", lw=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="sub-09")
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--mlp_cos_ckpt", required=True)
    parser.add_argument("--brain_ckpt", required=True)
    parser.add_argument("--vision_backbone", default="RN50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = build_config(args.subjects, args.vision_backbone)

    # Load brain encoder
    brain = instantiate_from_config(config["models"]["brain"]).to(device)
    bc = torch.load(args.brain_ckpt, map_location="cpu")
    bs = bc.get("state_dict", bc)
    brain.load_state_dict(strip_prefix(bs, "brain") or bs, strict=False)
    brain.eval()

    # Load diffusion
    diffusion = DiffusionModel(img_feat_dim=config["z_dim"], eeg_feat_dim=config["z_dim"], timesteps=1000).to(device)
    dc = torch.load(args.diffusion_ckpt, map_location="cpu")
    ds = dc.get("state_dict", dc)
    diffusion.load_state_dict(strip_prefix(ds, "diffusion") or ds, strict=False)
    diffusion.eval()

    # Load MLP-Cos generator
    mlp_cos = MLPGenerator(in_dim=config["z_dim"], out_dim=config["z_dim"]).to(device)
    mc = torch.load(args.mlp_cos_ckpt, map_location="cpu")
    # MLP checkpoint saved by train_mlp_baseline is the raw state_dict
    mlp_cos.load_state_dict(mc if "state_dict" not in mc else mc["state_dict"], strict=False)
    mlp_cos.eval()

    features = collect_all_features(config, brain, diffusion, mlp_cos, device)
    projected = pca_project(features)
    summary = compute_summary(features)

    out = args.out or f"assets/pca_mlp_vs_diffusion_{args.subjects}.png"
    plot_pca(projected, features, args.subjects, out, summary)


if __name__ == "__main__":
    main()
