import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime
import wandb
import yaml
import argparse

# 引入你的新模型
from torch.optim.lr_scheduler import CosineAnnealingLR
from motor_vae import PolicySkillTreeVAE, StandardScaler, MotionWindowDataset, get_amass_splits

# =========================
# 1. 工具类: Normalization & MPJPE
# =========================
def compute_mpjpe(pred_state_norm, target_state_norm, scaler):
    """
    计算 MPJPE (Mean Per Joint Position Error)
    必须先反归一化，才能得到真实的物理距离
    """
    # 1. Denormalize
    pred_state = scaler.inverse_transform_state(pred_state_norm)
    target_state = scaler.inverse_transform_state(target_state_norm)

    # 2. Extract Positions (Indices 8 to 74 based on smpl_to_d272)
    # 8 + 22 * 3 = 74
    # Shape: [B, T, 272]
    pos_idx_start = 8
    pos_idx_end = 8 + 66  # 22 joints * 3

    pred_pos = pred_state[..., pos_idx_start:pos_idx_end]
    target_pos = target_state[..., pos_idx_start:pos_idx_end]

    # 3. Reshape to [B, T, 22, 3]
    B, T, _ = pred_pos.shape
    pred_pos = pred_pos.view(B, T, 22, 3)
    target_pos = target_pos.view(B, T, 22, 3)

    # 4. Euclidean Distance per joint
    diff = torch.norm(pred_pos - target_pos, dim=-1)  # [B, T, 22]

    # 5. Mean over batch, time, joints (result in Meters)
    mpjpe = diff.mean()

    return mpjpe.item() * 1000.0  # Convert to mm


# =========================
# 2. Loss & Dataset (复用部分)
# =========================

class OptimalGaussianReconstructionLoss(nn.Module):
    def __init__(self, mode: str = "per_dim", min_log_sigma: float = -2.0):
        super().__init__()
        assert mode in ("global", "per_dim")
        self.mode = mode
        self.min_log_sigma = min_log_sigma

    @staticmethod
    def softclip(tensor: torch.Tensor, min_val: float) -> torch.Tensor:
        return min_val + F.softplus(tensor - min_val)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        if self.mode == "global":
            mse_mean = diff.pow(2).mean()
            log_sigma = 0.5 * torch.log(mse_mean + 1e-8)
            log_sigma = self.softclip(log_sigma, self.min_log_sigma)
        else:
            mse_mean = diff.pow(2).mean(dim=(0, 1), keepdim=True)
            log_sigma = 0.5 * torch.log(mse_mean + 1e-8)
            log_sigma = self.softclip(log_sigma, self.min_log_sigma)

        sigma_inv = torch.exp(-log_sigma)
        nll = 0.5 * (diff * sigma_inv) ** 2 + log_sigma
        return nll.mean()


def evaluate(model, data_loader, device, scaler, state_loss_fn, lambda_state):
    model.eval()
    stats = {"loss": 0., "rec_s": 0., "kl": 0., "mpjpe": 0.}
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            state_raw = batch["state"].to(device)

            # 1. Normalize
            state = scaler.transform_state(state_raw)

            # 2. Forward
            recon_state, mu, logvar = model(state)

            # 3. Loss (略...)
            if state_loss_fn is None:
                l_s = F.mse_loss(recon_state, state)
            else:
                l_s = state_loss_fn(recon_state, state)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state.shape[0]
            loss = lambda_state * l_s + 0.001 * l_kl

            # 4. MPJPE (略...)
            mpjpe = compute_mpjpe(recon_state, state, scaler)

            stats["loss"] += loss.item()
            stats["rec_s"] += l_s.item()
            stats["kl"] += l_kl.item()
            stats["mpjpe"] += mpjpe

            count += 1
            if count >= 50: break

    for k in stats:
        stats[k] /= count

    return stats


def get_kl_weight(step, total_steps, target_weight, start_weight=1e-4):
    """
    Scheme C: 从一个很小的非零值开始 Annealing
    避免模型在初期过度拟合 (Over-confident sigma)
    """
    if total_steps <= 0:
        return target_weight

    # 计算进度 0.0 -> 1.0
    ratio = min(1.0, step / total_steps)

    # 线性插值: start -> target
    current = start_weight + (target_weight - start_weight) * ratio
    return current
# =========================
# 4. 训练主流程
# =========================

def train_vae(config):
    # 解包配置
    data_root = config['data_root']
    test_data_root = config['test_data_root']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    window_size = config['window_size']
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    down_t = config.get('down_t', 3)  # 增加 down_t，默认 4
    norm_type = config.get('norm_type', 'LN')  # 增加 norm

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = float(config['lr'])  # 确保是浮点
    kl_anneal_steps = config['kl_anneal_steps']

    state_recons_type = config['state_recons_type']
    action_recons_type = config['action_recons_type']

    device = config.get('device', 'cuda')
    exp_name = config.get('exp_name', 'vae_run')
    wandb_project = config.get('wandb_project', 'VideoSkills-VAE')

    # 生成保存路径
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    save_dir = os.path.join(config['save_dir_root'], f"{exp_name}_{timestamp}")
    save_interval = config.get('save_interval', 5)

    # 1. WandB Init
    wandb.init(
        project=wandb_project,
        name=f"{exp_name}_{timestamp}",
        config=config  # 直接传入整个 config 字典
    )

    # all_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
    train_files, _, _ = get_amass_splits(data_root)
    _, _, test_files = get_amass_splits(test_data_root)

    train_ds = MotionWindowDataset(window_size, 2, action_dim, train_files)
    test_ds = MotionWindowDataset(window_size, 4, action_dim, test_files)  # Stride可以大一点用于eval

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. 计算 Normalization Stats
    scaler = StandardScaler(device=device)
    # 我们可以用一个小的数据集子集来计算 mean/std 以节省时间
    scaler.fit(DataLoader(train_ds, batch_size=4096, shuffle=False))  # 注意这里 fit 最好不要 shuffle
    scaler.to(device)

    # 3. 初始化模型
    model = PolicySkillTreeVAE(
        state_dim=state_dim,
        window_size=window_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        down_t=down_t,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ================= 4. Loss 配置 =================
    print(f"[Info] State Loss: {state_recons_type}")
    if state_recons_type == "optimal_gaussian":
        # State 相对平滑，允许 sigma 小一点 (-2.0)
        state_loss_fn = OptimalGaussianReconstructionLoss(mode="per_dim", min_log_sigma=-0.5).to(device)
    else:
        state_loss_fn = None # 使用 MSE

    print(f"[Info] Action Loss: {action_recons_type}")
    if action_recons_type == "optimal_gaussian":
        # ★★★ Action 比较抖动，建议把下限调高到 -1.0，防止过度自信导致的爆炸 ★★★
        action_loss_fn = OptimalGaussianReconstructionLoss(mode="per_dim", min_log_sigma=-0.5).to(device)
    else:
        action_loss_fn = None # 使用 MSE

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    torch.save({
        "mean_state": scaler.mean_state,
        "std_state": scaler.std_state,
        "mean_action": scaler.mean_action,
        "std_action": scaler.std_action
    }, os.path.join(save_dir, "scaler_stats.pt"))

    # 5. Training Loop
    best_mpjpe = float("inf")
    lambda_state = 1.0
    target_kl_weight = config.get('target_kl_weight')
    global_step = 0  # 用于内部计算 KL Annealing，不用于绘图轴

    for epoch in range(1, num_epochs + 1):
        model.train()

        # --- 初始化累积变量 ---
        epoch_loss = 0.0
        epoch_rec_s = 0.0
        epoch_kl = 0.0

        for batch in train_loader:
            state_raw = batch["state"].to(device)
            action_raw = batch["action"].to(device)
            global_step += 1

            # Normalize
            state = scaler.transform_state(state_raw)

            # Forward
            recon_state, mu, logvar = model(state)

            # --- 计算 State Loss ---
            if state_loss_fn is None:
                l_s = F.mse_loss(recon_state, state)
            else:
                l_s = state_loss_fn(recon_state, state)

            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state.shape[0]

            current_kl_weight = get_kl_weight(global_step, kl_anneal_steps, target_kl_weight)

            loss = lambda_state * l_s + current_kl_weight * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # --- 累积误差 (Accumulate) ---
            epoch_loss += loss.item()
            epoch_rec_s += l_s.item()
            epoch_kl += l_kl.item()

        # --- Epoch 结束: 计算平均值 ---
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_rec_s = epoch_rec_s / n_batches
        avg_kl = epoch_kl / n_batches

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 准备 Log 数据 ---
        log_dict = {
            "Train/Loss": avg_loss,
            "Train/RecS": avg_rec_s,
            "Train/KL": avg_kl,
            "Train/KL_Weight": current_kl_weight,
            "Train/LR": current_lr,
            "epoch": epoch  # 显式记录 epoch
        }

        # --- Evaluate (每 5 个 Epoch) ---
        if epoch % 5 == 0 or epoch == 1:
            eval_stats = evaluate(model, test_loader, device, scaler, state_loss_fn, lambda_state)

            # 判断是否 Best
            is_best = eval_stats['mpjpe'] < best_mpjpe
            if is_best:
                best_mpjpe = eval_stats['mpjpe']
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

            # ================= 打印 =================
            print("-" * 65)
            print(f"[{exp_name}] Epoch {epoch:03d}/{num_epochs} | Step {global_step} | LR {current_lr:.2e}")
            print(f"Train | Loss: {avg_loss:.4f} | KL_w: {current_kl_weight:.4f}")
            print(f"Eval  | MPJPE: {eval_stats['mpjpe']:.2f} mm {'[New Best!]' if is_best else ''}")
            print(f"      | Loss  : RecS={eval_stats['rec_s']:.4f} | KL={eval_stats['kl']:.4f}")
            print("-" * 65)
            # =================================================

            # Wandb Log (合并 Train 和 Eval)
            log_dict.update({
                "Eval/MPJPE_mm": eval_stats['mpjpe'],
                "Eval/RecS": eval_stats['rec_s'],
                "Eval/KL": eval_stats['kl'],
            })

        else:
            # 平时只打印一行简报
            print(f"[{exp_name}] Epoch {epoch:03d} | Loss: {avg_loss:.4f} | KL_w: {current_kl_weight:.4f}")

        # --- ★★★ Wandb Log (Per Epoch) ★★★ ---
        # 关键点：设置 step=epoch，这样横坐标就是 1, 2, 3...
        wandb.log(log_dict, step=epoch)

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MotorVAE with Config")
    parser.add_argument('--config', type=str, default='motor_vae/configs/default.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    # 1. 读取 YAML
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 设置随机种子 (可选)
    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"[Info] Loaded configuration from {args.config}")

    # 3. 开始训练
    train_vae(config)