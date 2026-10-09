#!/usr/bin/env python3
"""
Visualize real EEG features, generated EEG features, and image CLIP features.

This script does not modify training code. It loads a trained Lightning
checkpoint, extracts test-set features, projects them to 2D with t-SNE/UMAP/PCA,
and saves a publication-ready distribution plot.

Example:
  python scripts/plot_feature_space_distribution.py \
    --config configs/eeg/ubp.yaml \
    --checkpoint exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-09_seed0_genTrue/checkpoints/last.ckpt \
    --subjects sub-09 \
    --brain_backbone EEGProjectLayer \
    --vision_backbone RN50 \
    --method tsne
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.data_eeg import EEGDataset
from base.diffusion import DiffusionModel
from base.utils import instantiate_from_config


PRETRAIN_Z_DIM = {
    "RN50": 1024,
    "RN101": 512,
    "ViT-B-16": 512,
    "ViT-B-32": 512,
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
    "ViT-g-14": 1024,
    "ViT-bigG-14": 1280,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eeg/ubp.yaml")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path")
    parser.add_argument("--subjects", default="sub-09", help="One subject, e.g. sub-09")
    parser.add_argument("--brain_backbone", default="EEGProjectLayer")
    parser.add_argument("--vision_backbone", default="RN50")
    parser.add_argument("--dataset", default="eeg", choices=["eeg"])
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap", "pca"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=int, default=6, help="UBP blur kernel shift, same as main.py --c")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to assets/feature_space_<subject>_<method>.png",
    )
    parser.add_argument("--pdf", action="store_true", help="Also export PDF")
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save extracted high-dimensional features beside the figure",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    config = OmegaConf.load(args.config)
    config["dataset"] = args.dataset
    config["subjects"] = args.subjects
    config["exp_setting"] = "intra-subject"
    config["brain_backbone"] = args.brain_backbone
    config["vision_backbone"] = args.vision_backbone
    config["z_dim"] = PRETRAIN_Z_DIM[args.vision_backbone]
    config["c"] = args.c
    config["data"]["subjects"] = [args.subjects]
    config["data"]["model_type"] = args.vision_backbone
    config["data"]["test_batch_size"] = args.batch_size
    config["generative_encoding"] = True
    return config


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix_dot = prefix + "."
    return {k[len(prefix_dot):]: v for k, v in state_dict.items() if k.startswith(prefix_dot)}


def load_models(config, checkpoint_path: str, device: torch.device):
    brain = instantiate_from_config(config["models"]["brain"]).to(device)
    diffusion = DiffusionModel(
        img_feat_dim=config["z_dim"],
        eeg_feat_dim=config["z_dim"],
        timesteps=1000,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    brain_state = strip_state_dict_prefix(state_dict, "brain")
    diffusion_state = strip_state_dict_prefix(state_dict, "diffusion")

    missing_brain, unexpected_brain = brain.load_state_dict(brain_state, strict=False)
    missing_diff, unexpected_diff = diffusion.load_state_dict(diffusion_state, strict=False)

    if missing_brain or unexpected_brain:
        print("[warn] brain state mismatch")
        print("  missing:", missing_brain)
        print("  unexpected:", unexpected_brain)
    if missing_diff or unexpected_diff:
        print("[warn] diffusion state mismatch")
        print("  missing:", missing_diff)
        print("  unexpected:", unexpected_diff)

    brain.eval()
    diffusion.eval()
    return brain, diffusion


@torch.no_grad()
def collect_features(
    config,
    brain: torch.nn.Module,
    diffusion: torch.nn.Module,
    device: torch.device,
    max_samples: int,
) -> Dict[str, np.ndarray]:
    dataset = EEGDataset(config, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["test_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    real_eeg_features: List[torch.Tensor] = []
    generated_eeg_features: List[torch.Tensor] = []
    image_features: List[torch.Tensor] = []

    seen = 0
    for batch in loader:
        eeg = batch["eeg"].to(device).float()
        img_z = batch["img_features"].to(device).float()
        img_z = F.normalize(img_z, dim=-1)

        eeg_z = brain(eeg)
        eeg_z = F.normalize(eeg_z, dim=-1)

        gen_eeg_z = diffusion(img_z)
        gen_eeg_z = F.normalize(gen_eeg_z, dim=-1)

        real_eeg_features.append(eeg_z.cpu())
        generated_eeg_features.append(gen_eeg_z.cpu())
        image_features.append(img_z.cpu())

        seen += eeg.shape[0]
        if seen >= max_samples:
            break

    real_eeg = torch.cat(real_eeg_features, dim=0)[:max_samples].numpy()
    gen_eeg = torch.cat(generated_eeg_features, dim=0)[:max_samples].numpy()
    img = torch.cat(image_features, dim=0)[:max_samples].numpy()

    return {
        "real_eeg": real_eeg,
        "generated_eeg": gen_eeg,
        "image_clip": img,
    }


def project_features(features: Dict[str, np.ndarray], method: str, seed: int) -> Dict[str, np.ndarray]:
    labels = list(features.keys())
    sizes = [features[k].shape[0] for k in labels]
    x = np.concatenate([features[k] for k in labels], axis=0)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for --method tsne. "
                "Install scikit-learn or use --method pca."
            ) from exc

        perplexity = min(30, max(5, (x.shape[0] - 1) // 10))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        x_2d = reducer.fit_transform(x)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP is required for --method umap. Install umap-learn or use --method pca."
            ) from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.15,
            metric="cosine",
            random_state=seed,
        )
        x_2d = reducer.fit_transform(x)
    else:
        x_centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        x_2d = x_centered @ vt[:2].T

    out: Dict[str, np.ndarray] = {}
    start = 0
    for label, size in zip(labels, sizes):
        out[label] = x_2d[start:start + size]
        start += size
    return out


def pairwise_summary(features: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    real = features["real_eeg"]
    gen = features["generated_eeg"]
    img = features["image_clip"]

    gen_real = gen @ real.T
    gen_img = gen @ img.T
    off_diag_mask = ~np.eye(gen_real.shape[0], dtype=bool)

    return (
        float(np.diag(gen_real).mean()),
        float(gen_real[off_diag_mask].mean()),
        float(np.diag(gen_img).mean()),
    )


def plot_projection(
    projected: Dict[str, np.ndarray],
    features: Dict[str, np.ndarray],
    subject: str,
    method: str,
    out_path: Path,
    export_pdf: bool,
) -> None:
    real_diag, real_off, img_diag = pairwise_summary(features)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    style = {
        "image_clip": dict(label="Image CLIP feature", color="#377eb8", marker="x", s=46, alpha=0.72),
        "real_eeg": dict(label="Real EEG feature", color="#2ca25f", marker="o", s=38, alpha=0.72),
        "generated_eeg": dict(label="Generated EEG feature", color="#8e44ad", marker="^", s=42, alpha=0.78),
    }

    for key in ["image_clip", "real_eeg", "generated_eeg"]:
        points = projected[key]
        ax.scatter(points[:, 0], points[:, 1], edgecolors="none", **style[key])

    ax.set_title(f"Feature-space distribution ({subject}, {method.upper()})", weight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(color="#d9d9d9", lw=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    summary = (
        f"cos(gen, real) diag = {real_diag:.3f}\n"
        f"cos(gen, real) off-diag = {real_off:.3f}\n"
        f"cos(gen, image) diag = {img_diag:.3f}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#bdbdbd", alpha=0.92, boxstyle="round,pad=0.35"),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if export_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    print(f"Saved figure to {out_path}")
    if export_pdf:
        print(f"Saved PDF to {out_path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    config = build_config(args)
    brain, diffusion = load_models(config, args.checkpoint, device)
    features = collect_features(config, brain, diffusion, device, args.max_samples)
    projected = project_features(features, args.method, args.seed)

    out_path = Path(args.out) if args.out else Path(
        f"assets/feature_space_{args.subjects}_{args.method}.png"
    )
    plot_projection(projected, features, args.subjects, args.method, out_path, args.pdf)

    if args.save_npz:
        npz_path = out_path.with_suffix(".npz")
        np.savez(
            npz_path,
            real_eeg=features["real_eeg"],
            generated_eeg=features["generated_eeg"],
            image_clip=features["image_clip"],
            real_eeg_2d=projected["real_eeg"],
            generated_eeg_2d=projected["generated_eeg"],
            image_clip_2d=projected["image_clip"],
        )
        print(f"Saved features to {npz_path}")


if __name__ == "__main__":
    main()
