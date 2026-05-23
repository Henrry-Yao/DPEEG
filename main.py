import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F


##import user lib
from base.diffusion import DiffusionModel


from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device

device = get_device('auto')

def load_model(config,train_loader,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model,config,train_loader,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,train_loader,test_loader, model_type = 'RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []
    
        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3
        
        self.mAP_total = 0
        # 添加diffusion模型用于生成式编码
        self.match_similarities = []

        # 添加diffusion模型用于生成式编码
        self.diffusion = DiffusionModel(
            img_feat_dim=self.z_dim,
            eeg_feat_dim=self.z_dim,
            timesteps=1000
        )
        self.diffusion_criterion = torch.nn.MSELoss()

        self.subject_sim_history = {}
        self.subject_mu = {}
        self.subject_sigma = {}

        # 双向相似度记录
        self.gen_img_similarities = []
        self.gen_eeg_similarities = []
        self.gen_eeg_off_diag_similarities = []

    def forward(self, batch,sample_posterior=False):
 
        idx = batch['idx'].cpu().detach().numpy() 
        eeg = batch['eeg']
        img = batch['img']

        img_z  = batch['img_features']
        
        eeg_z = self.brain(eeg)
        img_z = img_z/img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]
            self.sim[idx] = batch_sim

            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            # 个体化不确定性：按受试者分组（向量化版本）
            subjects = batch['subject'].cpu().numpy()
            unique_subs = np.unique(subjects)

            for sub in unique_subs:
                sub_key = str(int(sub))
                sub_mask = (subjects == sub)
                sub_sims = batch_sim[sub_mask]

                if sub_key not in self.subject_sim_history:
                    self.subject_sim_history[sub_key] = []
                self.subject_sim_history[sub_key].extend(sub_sims.tolist())

                history = self.subject_sim_history[sub_key]
                if len(history) >= 10:
                    mu = np.mean(history)
                    sigma = max(np.std(history, ddof=1), 1e-8)
                else:
                    mu = np.mean(batch_sim)
                    sigma = max(np.std(batch_sim, ddof=1) if len(batch_sim) > 1 else 1.0, 1e-8)

                lower_bound = mu - z_alpha_2 * sigma
                upper_bound = mu + z_alpha_2 * sigma

                sub_diag = diagonal_elements[sub_mask]
                match_label[sub_mask & (sub_diag > upper_bound)] = 0
                match_label[sub_mask & (sub_diag < lower_bound)] = 2

            self.match_label[idx] = match_label
            loss = total_loss
        else:
            loss = total_loss
        return eeg_z, img_z, loss
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, base_loss = self(batch, sample_posterior=True)

        warmup_epochs = self.config['train'].get('warmup_epochs', 30)
        current_epoch = self.current_epoch

        if self.config.get('generative_encoding', False):
            if current_epoch < warmup_epochs:
                # 阶段一：只训练EEG编码器，扩散模型不参与
                total_loss = base_loss
            else:
                # 阶段二：detach阻断梯度，只训练扩散模型
                eeg_z_d = eeg_z.detach()
                img_z_d = img_z.detach()
                noise_pred, noise = self.diffusion(img_z_d, eeg_z_d)
                diffusion_loss = self.diffusion_criterion(noise_pred, noise)
                total_loss = diffusion_loss
                self.log('diffusion_loss', diffusion_loss, on_step=True,
                        prog_bar=True, batch_size=batch_size)
        else:
            total_loss = base_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=batch_size)

        with torch.no_grad():
            eeg_z_norm = eeg_z.detach() / (eeg_z.detach().norm(dim=-1, keepdim=True) + 1e-8)
            img_z_norm = img_z.detach() / (img_z.detach().norm(dim=-1, keepdim=True) + 1e-8)
            similarity = (eeg_z_norm @ img_z_norm.T)
            top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
            self.all_predicted_classes.append(top_k_indices.cpu().numpy())
            label = torch.arange(0, batch_size).to(self.device)
            self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_correct = all_predicted_classes[:, 0] == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []
            counter = Counter(self.match_label)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: counter[k] for k in key_mapping}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True,
                        logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label

        return total_loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
    
        eeg_z, img_z, loss= self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        if self.config.get('generative_encoding', False):
            generated_eeg_z = self.diffusion(img_z)

            # 归一化
            generated_eeg_z = generated_eeg_z / (
                generated_eeg_z.norm(dim=-1, keepdim=True) + 1e-8
            )
            eeg_z_norm = eeg_z / (eeg_z.norm(dim=-1, keepdim=True) + 1e-8)
            # img_z 已在 forward 里归一化，直接用

            # 1. 生成EEG vs 图像：语义一致性（用于检索评估）
            gen_img_similarity = (generated_eeg_z @ img_z.T)

            # 2. 生成EEG vs 真实EEG：信号真实性
            gen_eeg_similarity = (generated_eeg_z @ eeg_z_norm.T)

            # 记录对角线相似度
            self.gen_img_similarities.extend(
                gen_img_similarity.diag().detach().cpu().tolist()
            )
            self.gen_eeg_similarities.extend(
                gen_eeg_similarity.diag().detach().cpu().tolist()
            )

            # 非对角线相似度
            off_diag_mask = ~torch.eye(
                batch_size, dtype=torch.bool, device=self.device
            )
            off_diag_sim = gen_eeg_similarity[off_diag_mask].mean().item()
            self.gen_eeg_off_diag_similarities.append(off_diag_sim)

            # 检索评估（保留原有逻辑）
            top_kvalues, top_k_indices = gen_img_similarity.topk(5, dim=-1)
            self.all_predicted_classes.append(top_k_indices.cpu().numpy())
            label = torch.arange(0, batch_size).to(self.device)
            self.all_true_labels.extend(label.cpu().numpy())
            self.match_similarities.extend(
                gen_img_similarity.diag().detach().cpu().tolist()
            )
            for i in range(gen_img_similarity.shape[0]):
                rank = (torch.argsort(-gen_img_similarity[i]) == i).nonzero()[0][0] + 1
                self.mAP_total += (1 / rank)
        else:
            eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
            similarity = (eeg_z @ img_z.T)
            top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
            self.all_predicted_classes.append(top_k_indices.cpu().numpy())
            label = torch.arange(0, batch_size).to(self.device)
            self.all_true_labels.extend(label.cpu().numpy())
            self.match_similarities.extend(
                similarity.diag().detach().cpu().tolist()
            )
            for i in range(similarity.shape[0]):
                sims = similarity[i, :]
                rank = (torch.argsort(-sims) == i).nonzero()[0][0] + 1
                self.mAP_total += (1 / rank)

        return loss
        
    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        # 双向相似度汇总输出
        if self.gen_img_similarities:
            avg_gen_img_sim = np.mean(self.gen_img_similarities)
            avg_gen_eeg_sim = np.mean(self.gen_eeg_similarities)
            avg_off_diag    = np.mean(self.gen_eeg_off_diag_similarities)
            self.log('gen_img_cosine_sim', avg_gen_img_sim, sync_dist=True)
            self.log('gen_eeg_cosine_sim', avg_gen_eeg_sim, sync_dist=True)
            self.log('gen_eeg_off_diag_sim', avg_off_diag,  sync_dist=True)
            print(f"生成EEG vs 图像相似度:           {avg_gen_img_sim:.4f}")
            print(f"生成EEG vs 真实EEG相似度:        {avg_gen_eeg_sim:.4f}")
            print(f"生成EEG vs 非对应真实EEG相似度:  {avg_off_diag:.4f}")

        # 清空
        self.gen_img_similarities = []
        self.gen_eeg_similarities = []
        self.gen_eeg_off_diag_similarities = []

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item(),'mAP':self.mAP,'similarity':self.match_similarities}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](
            self.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=1e-4
        )
        return [optimizer]
    
    def on_train_epoch_start(self):
        if not self.config.get('generative_encoding', False):
            return
        warmup_epochs = self.config['train'].get('warmup_epochs', 30)
        if self.current_epoch == warmup_epochs:
            print(f"\n[Epoch {self.current_epoch}] 切换到阶段二：开始训练扩散模型")
        elif self.current_epoch == 0:
            print(f"\n[Epoch 0] 阶段一：训练EEG编码器，共 {warmup_epochs} epochs")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )
    parser.add_argument(
    "--generative_encoding",
    action="store_true",
    default=False,
    help="enable generative encoding with diffusion model",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'],exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}_gen{opt.generative_encoding}")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)

    print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,     
            patience=15, 
            verbose=False,
            mode='max' 
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=120,
            verbose=False,  
            mode='min' 
        )

    trainer = Trainer(
    log_every_n_steps=10,
    strategy=DDPStrategy(find_unused_parameters=True),  # ← True
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=opt.epoch,
    devices=[device],
    accelerator='cuda',
    logger=logger
    )
    print(trainer.logger.log_dir)

    ckpt_path = 'last' #None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=ckpt_path)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__=="__main__":
    main()