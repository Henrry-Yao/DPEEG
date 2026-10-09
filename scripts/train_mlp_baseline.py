#!/usr/bin/env python3
"""
Train MLP baselines for generative neural encoding.
- MLP-MSE: img_z → MLP → pred_eeg_z, trained with MSE loss
- MLP-Cos: img_z → MLP → pred_eeg_z, trained with Cosine similarity loss

Uses frozen pretrained EEG encoder as target. Evaluates Top-1 + cosine metrics.
"""
import argparse, os, sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base.data_eeg import EEGDataset
from base.utils import instantiate_from_config
from omegaconf import OmegaConf


# ── MLP Generator ─────────────────────────────────────────────────
class MLPGenerator(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, hidden_dims=None, dropout=0.15):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── Training / Eval Loop ─────────────────────────────────────────
def run_experiment(config, brain_ckpt_path, loss_type, device, out_dir):
    """
    loss_type: 'mse' or 'cosine'
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Data ──
    test_dataset = EEGDataset(config, mode='test')
    train_dataset = EEGDataset(config, mode='train')
    test_loader  = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

    # ── EEG encoder (frozen) ──
    brain = instantiate_from_config(config['models']['brain']).to(device)
    ckpt = torch.load(brain_ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    # strip 'brain.' prefix if present
    brain_state = {k.replace('brain.', ''): v for k, v in state_dict.items() if k.startswith('brain.')}
    if not brain_state:
        brain_state = state_dict
    brain.load_state_dict(brain_state, strict=False)
    brain.eval()
    for p in brain.parameters():
        p.requires_grad = False

    # ── MLP generator ──
    mlp = MLPGenerator(in_dim=config['z_dim'], out_dim=config['z_dim']).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-4)

    if loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        def cosine_loss(pred, target):
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)
            return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
        criterion = cosine_loss

    # ── Training ──
    n_epochs = 50
    print("Training MLP-%s for %d epochs ..." % (loss_type.upper(), n_epochs))
    for epoch in range(n_epochs):
        mlp.train()
        total_loss = 0.0
        for batch in train_loader:
            img_z = batch['img_features'].to(device).float()
            eeg_raw = batch['eeg'].to(device).float()

            with torch.no_grad():
                eeg_z = brain(eeg_raw)
                eeg_z = F.normalize(eeg_z, dim=-1)

            pred = mlp(img_z)
            loss = criterion(pred, eeg_z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print("  Epoch %3d/%d  loss=%.4f" % (epoch + 1, n_epochs, total_loss / len(train_loader)))

    # ── Save generator checkpoint (loaded by the analysis scripts) ──
    torch.save(mlp.state_dict(), os.path.join(out_dir, "model.pt"))

    # ── Evaluation ──
    mlp.eval()
    all_gen_eeg = []
    all_real_eeg = []
    all_img_z = []

    with torch.no_grad():
        for batch in test_loader:
            img_z = batch['img_features'].to(device).float()
            eeg_raw = batch['eeg'].to(device).float()

            eeg_z = brain(eeg_raw)
            eeg_z = F.normalize(eeg_z, dim=-1)
            gen_z = mlp(img_z)
            gen_z = F.normalize(gen_z, dim=-1)

            all_gen_eeg.append(gen_z.cpu())
            all_real_eeg.append(eeg_z.cpu())
            all_img_z.append(img_z.cpu())

    gen_eeg = torch.cat(all_gen_eeg, dim=0)  # (200, 1024)
    real_eeg = torch.cat(all_real_eeg, dim=0)
    img_z = torch.cat(all_img_z, dim=0)
    N = gen_eeg.shape[0]

    # ── Retrieval Top-1 (生成EEG → 检索图像 = gen @ img.T) ──
    similarity = gen_eeg @ img_z.T  # (200, 200)
    top_preds = similarity.argmax(dim=-1)
    labels = torch.arange(N)
    top1_acc = (top_preds == labels).float().mean().item() * 100

    # ── Cosine metrics ──
    gen_real_sim = (gen_eeg @ real_eeg.T).numpy()  # (200, 200)
    diag_mask = np.eye(N, dtype=bool)
    off_mask = ~diag_mask

    diag = float(gen_real_sim[diag_mask].mean())
    off  = float(gen_real_sim[off_mask].mean())
    diag_off_ratio = diag / (off + 1e-8)

    # ── Also compute gen_img_cosine ──
    gen_img_sim_diag = float((gen_eeg * img_z).sum(dim=-1).mean())

    results = {
        'model': 'MLP-%s' % loss_type.upper(),
        'top1': round(top1_acc, 2),
        'diag': round(diag, 4),
        'off': round(off, 4),
        'diag_off_ratio': round(diag_off_ratio, 2),
        'gen_img_sim': round(gen_img_sim_diag, 4),
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("Results: %s" % json.dumps(results))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eeg/ubp.yaml')
    parser.add_argument('--subjects', default='sub-09')
    parser.add_argument('--brain_backbone', default='EEGProjectLayer')
    parser.add_argument('--vision_backbone', default='RN50')
    parser.add_argument('--brain_ckpt', required=True,
                        help='Path to pretrained EEG encoder checkpoint (genFalse)')
    parser.add_argument('--loss', default='mse', choices=['mse', 'cosine'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = OmegaConf.load(args.config)
    config['dataset'] = 'eeg'
    config['exp_setting'] = 'intra-subject'
    config['brain_backbone'] = args.brain_backbone
    config['vision_backbone'] = args.vision_backbone
    config['data']['subjects'] = [args.subjects]
    config['data']['model_type'] = args.vision_backbone
    config['data']['test_batch_size'] = 200
    config['data']['train_batch_size'] = 1024
    config['data']['train_avg'] = True
    config['data']['test_avg'] = True
    config['data']['uncertainty_aware'] = True

    pretrain_z = {
        'RN50': 1024, 'RN101': 512, 'ViT-B-16': 512, 'ViT-B-32': 512,
        'ViT-L-14': 768, 'ViT-H-14': 1024, 'ViT-g-14': 1024, 'ViT-bigG-14': 1280
    }
    config['z_dim'] = pretrain_z[args.vision_backbone]
    config['c'] = 6

    out_dir = 'exp/mlp_baseline/%s_%s_%s_s%d' % (args.subjects, args.loss, args.vision_backbone, args.seed)
    run_experiment(config, args.brain_ckpt, args.loss, device, out_dir)


if __name__ == '__main__':
    main()
