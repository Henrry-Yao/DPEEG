import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers import DDPMScheduler


class DiffusionPriorImgEEG(nn.Module):
    """
    UNet结构的扩散先验：img_z → eeg_z
    x: eeg特征（加噪目标）
    c: img特征（条件）
    """
    def __init__(
            self,
            embed_dim=1024,   # eeg_z维度（生成目标）
            cond_dim=1024,    # img_z维度（条件）
            hidden_dim=None,
            time_embed_dim=1024,
            act_fn=nn.SiLU,
            dropout=0.2,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [2048, 1024, 512, 256, 128, 64]

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        # 时间步嵌入
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # UNet编码器
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )
        self.encode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i])
            for i in range(self.num_layers - 1)
        ])
        self.encode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i])
            for i in range(self.num_layers - 1)
        ])
        self.encode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.LayerNorm(hidden_dim[i + 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1)
        ])

        # UNet解码器
        self.decode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i])
            for i in range(self.num_layers - 1, 0, -1)
        ])
        self.decode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i])
            for i in range(self.num_layers - 1, 0, -1)
        ])
        self.decode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                nn.LayerNorm(hidden_dim[i - 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1, 0, -1)
        ])

        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, t, c=None):
        """
        x: (B, embed_dim) 加噪后的eeg特征
        t: (B,) 时间步
        c: (B, cond_dim) img特征条件，None时为无条件
        """
        t_proj = self.time_proj(t)  # (B, time_embed_dim)

        x = self.input_layer(x)

        # 编码器
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t_proj)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)

        # 解码器
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t_proj)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x = x + hidden_activations[-1 - i]  # 跳跃连接

        return self.output_layer(x)


class DiffusionModel(nn.Module):
    """
    封装DiffusionPriorImgEEG，提供与main.py兼容的接口。
    training: forward(img_z, eeg_z) → (noise_pred, noise)
    inference: forward(img_z) → generated_eeg_z
    """
    def __init__(self, img_feat_dim=1024, eeg_feat_dim=1024, timesteps=1000):
        super().__init__()
        self.eeg_feat_dim = eeg_feat_dim
        self.timesteps = timesteps

        self.prior = DiffusionPriorImgEEG(
            embed_dim=eeg_feat_dim,
            cond_dim=img_feat_dim,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        # 把scheduler的tensor注册为buffer，避免设备不匹配
        self.register_buffer(
            'alphas_cumprod',
            self.scheduler.alphas_cumprod.clone().detach().float()
        )

    def forward(self, img_feat, eeg_feat=None):
        """
        训练时：img_feat + eeg_feat → (noise_pred, noise)
        推理时：img_feat → generated_eeg_z
        """
        if eeg_feat is not None:
            # 训练模式：噪声预测
            B = eeg_feat.shape[0]
            t = torch.randint(0, self.timesteps, (B,), device=eeg_feat.device)
            noise = torch.randn_like(eeg_feat)

            # 用scheduler加噪
            noisy_eeg = self.scheduler.add_noise(eeg_feat, noise, t)
            noise_pred = self.prior(noisy_eeg, t, img_feat)
            return noise_pred, noise
        else:
            # 推理模式：CFG去噪生成
            return self._generate(img_feat)

    def _generate(self, img_feat, num_inference_steps=100, guidance_scale=5.0):
        B = img_feat.shape[0]
        device = img_feat.device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        x = torch.randn(B, self.eeg_feat_dim, device=device)

        for t in self.scheduler.timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # CFG：有条件 + 无条件
            noise_pred_cond   = self.prior(x, t_batch, img_feat)
            noise_pred_uncond = self.prior(x, t_batch, None)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # 逐样本step（scheduler不支持批量不同t）
            x_out = []
            for idx in range(B):
                step_out = self.scheduler.step(
                    noise_pred[idx:idx+1],
                    t_batch[idx].item(),
                    x[idx:idx+1]
                )
                x_out.append(step_out.prev_sample)
            x = torch.cat(x_out, dim=0)

        return F.normalize(x, dim=-1)