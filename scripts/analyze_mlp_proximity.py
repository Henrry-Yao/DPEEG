#!/usr/bin/env python3
"""Quantitative: how close are MLP-Cos vs Diffusion features to Image vs Real EEG?"""
import torch, torch.nn.functional as F, numpy as np, sys
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base.data_eeg import EEGDataset
from base.utils import instantiate_from_config
from base.diffusion import DiffusionModel
from scripts.train_mlp_baseline import MLPGenerator

def strip_prefix(d, p):
    return {k.replace(p + ".", ""): v for k, v in d.items() if k.startswith(p + ".")}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = OmegaConf.load("configs/eeg/ubp.yaml")
config.dataset = "eeg"
config.exp_setting = "intra-subject"
config.brain_backbone = "EEGProjectLayer"
config.vision_backbone = "RN50"
config.z_dim = 1024
config.c = 6
config.data.subjects = ["sub-09"]
config.data.model_type = "RN50"
config.data.test_batch_size = 200
config.data.train_avg = True
config.data.test_avg = True
config.data.uncertainty_aware = True

ds = EEGDataset(config, mode="test")
loader = DataLoader(ds, batch_size=200, shuffle=False, num_workers=0)

# Models
brain = instantiate_from_config(config.models.brain).to(device)
ckpt = torch.load("exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-09_seed0_genFalse/checkpoints/last.ckpt", map_location="cpu")
brain.load_state_dict(strip_prefix(ckpt.get("state_dict", ckpt), "brain"), strict=False)
brain.eval()

diffusion = DiffusionModel(1024, 1024, 1000).to(device)
dc = torch.load("exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-09_seed0_genTrue/checkpoints/last.ckpt", map_location="cpu")
diffusion.load_state_dict(strip_prefix(dc.get("state_dict", dc), "diffusion"), strict=False)
diffusion.eval()

mlp = MLPGenerator(1024, 1024).to(device)
mlp.load_state_dict(torch.load("exp/mlp_baseline/sub-09_cosine_RN50/model.pt", map_location="cpu"))
mlp.eval()

# Collect
all_real, all_diff, all_mlp, all_img = [], [], [], []
for batch in loader:
    eeg = batch["eeg"].to(device).float()
    img_z = batch["img_features"].to(device).float()
    img_z = F.normalize(img_z, dim=-1)
    eeg_z = brain(eeg)
    eeg_z = F.normalize(eeg_z, dim=-1)
    diff_z = diffusion(img_z)
    diff_z = F.normalize(diff_z, dim=-1)
    mlp_z = mlp(img_z)
    mlp_z = F.normalize(mlp_z, dim=-1)
    all_real.append(eeg_z.cpu())
    all_diff.append(diff_z.cpu())
    all_mlp.append(mlp_z.cpu())
    all_img.append(img_z.cpu())

real = torch.cat(all_real).detach().numpy()
diff = torch.cat(all_diff).detach().numpy()
mlp_f = torch.cat(all_mlp).detach().numpy()
img = torch.cat(all_img).detach().numpy()
N = real.shape[0]

def euclidean_dist(a, centroid):
    return float(np.mean(np.linalg.norm(a - centroid[np.newaxis, :], axis=1)))

real_centroid = real.mean(axis=0)
img_centroid = img.mean(axis=0)

print("=" * 65)
print("Distance Analysis (sub-09)")
print("-" * 65)
print("MLP-Cos  -> Real EEG centroid:  %.4f" % euclidean_dist(mlp_f, real_centroid))
print("MLP-Cos  -> Image CLIP centroid: %.4f" % euclidean_dist(mlp_f, img_centroid))
print("Diff Gen -> Real EEG centroid:  %.4f" % euclidean_dist(diff, real_centroid))
print("Diff Gen -> Image CLIP centroid: %.4f" % euclidean_dist(diff, img_centroid))
print("Real EEG -> Real EEG centroid:  %.4f" % euclidean_dist(real, real_centroid))
print("Image    -> Image CLIP centroid: %.4f" % euclidean_dist(img, img_centroid))

print()
print("-" * 65)
print("Cosine similarity ratios")
print("-" * 65)
for name, gen in [("MLP-Cos  ", mlp_f), ("Diffusion", diff)]:
    sim_real = gen @ real.T
    sim_img = gen @ img.T
    diag_mask = np.eye(N, dtype=bool)
    off_mask = ~diag_mask
    d_real = float(sim_real[diag_mask].mean())
    off_real = float(sim_real[off_mask].mean())
    d_img = float(sim_img[diag_mask].mean())
    # Proximity ratio: >1 = closer to image, <1 = closer to real EEG
    prox_to_img_vs_real = d_img / (d_real + 1e-8)
    print("%s cos(gen,real)=%.4f  cos(gen,img)=%.4f  off_real=%.4f  img/real_ratio=%.2f" % (
        name, d_real, d_img, off_real, prox_to_img_vs_real))

print()
print("Interpretation: img/real_ratio > 1.0 => features lean toward IMAGE space")
print("                img/real_ratio < 1.0 => features lean toward REAL EEG space")
