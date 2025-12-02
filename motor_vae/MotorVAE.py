import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime
import wandb

# 引入你的新模型
from torch.optim.lr_scheduler import CosineAnnealingLR
from motor_vae import PolicySkillTreeVAE, StandardScaler, MotionWindowDataset

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


def compute_jerk(action_seq, dt=1 / 30.0):
    """
    计算动作序列的平滑度 (Jerk)
    action_seq: [B, T, D]
    返回: 平均 Jerk 模长
    """
    # 1. 速度 (一阶差分)
    vel = action_seq[:, 1:, :] - action_seq[:, :-1, :]

    # 2. 加速度 (二阶差分)
    acc = vel[:, 1:, :] - vel[:, :-1, :]

    # 3. 加加速度 (Jerk, 三阶差分) - 这里我们通常用二阶差分近似衡量平滑度
    # 但严格意义上的 Jerk 是三阶。
    # 在动作捕捉领域，通常直接看 "Acceleration Change" 或者 "Velocity Change"
    # 这里我们计算二阶差分的模长，作为平滑度指标 (Smoothness Cost)

    # 计算 L2 Norm
    smoothness = torch.norm(acc, dim=-1).mean()

    # 如果要严格的 Jerk (三阶):
    # jerk = acc[:, 1:, :] - acc[:, :-1, :]
    # return torch.norm(jerk, dim=-1).mean()

    return smoothness

def compute_r2_score(pred, target):
    # pred, target: [N, D]
    target_mean = torch.mean(target, dim=0, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=0)
    ss_res = torch.sum((target - pred) ** 2, dim=0)

    # 防止分母为 0
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return torch.mean(r2).item()  # 对所有维度取平均

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




# =========================
# 3. 评估循环
# =========================


def evaluate(model, data_loader, device, scaler, state_loss_fn, lambda_state, lambda_action):
    model.eval()
    stats = {
        "loss": 0., "rec_s": 0., "rec_a": 0., "kl": 0., "mpjpe": 0.,
        "r2_score": 0., "cosine_sim": 0., "mae": 0., "gt_jerk": 0., "pred_jerk": 0., "jerk_ratio": 0.
    }
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            state_raw = batch["state"].to(device)
            action_raw = batch["action"].to(device)

            # 1. Normalize
            state = scaler.transform_state(state_raw)
            action = scaler.transform_action(action_raw)

            # 2. Forward
            recon_state, pred_action, mu, logvar = model(state)

            # 3. Loss (略...)
            if state_loss_fn is None:
                l_s = F.mse_loss(recon_state, state)
            else:
                l_s = state_loss_fn(recon_state, state)
            l_a = F.mse_loss(pred_action, action)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state.shape[0]
            loss = lambda_state * l_s + lambda_action * l_a + 0.001 * l_kl

            # 4. MPJPE (略...)
            mpjpe = compute_mpjpe(recon_state, state, scaler)

            # ================= 新增指标计算 =================
            # 注意：建议在 Denormalized (原始物理空间) 上计算这些指标，物理意义更强
            # 但如果你只关心相对质量，Normalized 空间也可以。这里演示用原始空间：

            # 反归一化 Action
            pred_action_phys = scaler.inverse_transform_action(pred_action)
            action_raw_phys = scaler.inverse_transform_action(action) # 注意这里要用 batch['action'] 的反归一化
            # action_raw 是真实的物理 Action

            # ================= Jerk Analysis =================
            # 计算 Ground Truth 的抖动程度
            gt_smoothness = compute_jerk(action_raw_phys)

            # 计算 VAE 重建的抖动程度
            pred_smoothness = compute_jerk(pred_action_phys)

            # 计算比率 Ratio < 1.0 意味着变平滑了
            jerk_ratio = pred_smoothness / (gt_smoothness + 1e-6)

            stats["gt_jerk"] += gt_smoothness.item()
            stats["pred_jerk"] += pred_smoothness.item()
            stats["jerk_ratio"] += jerk_ratio.item()

            # A. R2 Score
            # Flatten batch and time: [B*T, D]
            flat_pred = pred_action_phys.reshape(-1, pred_action_phys.shape[-1])
            flat_target = action_raw.reshape(-1, action_raw.shape[-1])
            r2 = compute_r2_score(flat_pred, flat_target)

            # B. Cosine Similarity
            # 计算每一帧向量的相似度，然后取平均
            # dim=-1 表示沿着动作维度计算
            cos_sim = F.cosine_similarity(pred_action_phys, action_raw, dim=-1).mean().item()

            # C. MAE (Mean Absolute Error)
            mae = torch.abs(pred_action_phys - action_raw).mean().item()

            # ==============================================

            stats["loss"] += loss.item()
            stats["rec_s"] += l_s.item()
            stats["rec_a"] += l_a.item()
            stats["kl"] += l_kl.item()
            stats["mpjpe"] += mpjpe
            stats["r2_score"] += r2
            stats["cosine_sim"] += cos_sim
            stats["mae"] += mae

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

def train_vae(
        data_root,
        state_dim=272,
        action_dim=69,
        window_size=32,
        batch_size=2048,
        num_epochs=100,
        lr=3e-4,
        kl_anneal_steps = 5000,
        device="cuda",
        save_dir=None,
        state_recons_type="optimal_gaussian",
        action_recons_type="optimal_gaussian",
        wandb_project = "VideoSkills-VAE",  # <--- 新增 wandb 参数
        wandb_name = 'double_gaussian'
):
    # 1. 准备数据

    if wandb_name is None:
        wandb_name = f"vae_{datetime.datetime.now().strftime('%m%d_%H%M')}"

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "window_size": window_size,
            "batch_size": batch_size,
            "lr": lr,
            "kl_anneal_steps": kl_anneal_steps,
            "state_recons_type": state_recons_type,
            "lambda_state": 1.0,
            "lambda_action": 5.0
        }
    )

    all_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
    # 简单划分 Train/Test
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    train_ds = MotionWindowDataset(window_size, 8, action_dim, train_files)
    test_ds = MotionWindowDataset(window_size, 32, action_dim, test_files)  # Stride可以大一点用于eval

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. 计算 Normalization Stats
    scaler = StandardScaler(device=device)
    # 我们可以用一个小的数据集子集来计算 mean/std 以节省时间
    scaler.fit(DataLoader(train_ds, batch_size=4096, shuffle=False))  # 注意这里 fit 最好不要 shuffle
    scaler.to(device)

    # 3. 初始化模型
    model = PolicySkillTreeVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        window_size=window_size,
        hidden_dim=512,  # 模型内部参数
        latent_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ================= 4. Loss 配置 =================
    print(f"[Info] State Loss: {state_recons_type}")
    if state_recons_type == "optimal_gaussian":
        # State 相对平滑，允许 sigma 小一点 (-2.0)
        state_loss_fn = OptimalGaussianReconstructionLoss(mode="per_dim", min_log_sigma=-1.0).to(device)
    else:
        state_loss_fn = None # 使用 MSE

    print(f"[Info] Action Loss: {action_recons_type}")
    if action_recons_type == "optimal_gaussian":
        # ★★★ Action 比较抖动，建议把下限调高到 -1.0，防止过度自信导致的爆炸 ★★★
        action_loss_fn = OptimalGaussianReconstructionLoss(mode="per_dim", min_log_sigma=-1.0).to(device)
    else:
        action_loss_fn = None # 使用 MSE

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # 保存 Scaler stats，推理时需要
    if save_dir:
        torch.save({
            "mean_state": scaler.mean_state,
            "std_state": scaler.std_state,
            "mean_action": scaler.mean_action,
            "std_action": scaler.std_action
        }, os.path.join(save_dir, "scaler_stats.pt"))

    # 5. Training Loop
    best_mpjpe = float("inf")
    lambda_state = 1.0
    lambda_action = 5.0
    target_kl_weight = 0.01
    global_step = 0  # 用于内部计算 KL Annealing，不用于绘图轴

    for epoch in range(1, num_epochs + 1):
        model.train()

        # --- 初始化累积变量 ---
        epoch_loss = 0.0
        epoch_rec_s = 0.0
        epoch_rec_a = 0.0
        epoch_kl = 0.0

        for batch in train_loader:
            state_raw = batch["state"].to(device)
            action_raw = batch["action"].to(device)
            global_step += 1

            # Normalize
            state = scaler.transform_state(state_raw)
            action = scaler.transform_action(action_raw)

            # Forward
            recon_state, pred_action, mu, logvar = model(state)

            # --- 计算 State Loss ---
            if state_loss_fn is None:
                l_s = F.mse_loss(recon_state, state)
            else:
                l_s = state_loss_fn(recon_state, state)

            # --- 计算 Action Loss (修改处) ---
            if action_loss_fn is None:
                l_a = F.mse_loss(pred_action, action)
            else:
                l_a = action_loss_fn(pred_action, action)

            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state.shape[0]

            current_kl_weight = get_kl_weight(global_step, kl_anneal_steps, target_kl_weight)

            loss = lambda_state * l_s + lambda_action * l_a + current_kl_weight * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- 累积误差 (Accumulate) ---
            epoch_loss += loss.item()
            epoch_rec_s += l_s.item()
            epoch_rec_a += l_a.item()
            epoch_kl += l_kl.item()

        # --- Epoch 结束: 计算平均值 ---
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_rec_s = epoch_rec_s / n_batches
        avg_rec_a = epoch_rec_a / n_batches
        avg_kl = epoch_kl / n_batches

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 准备 Log 数据 ---
        log_dict = {
            "Train/Loss": avg_loss,
            "Train/RecS": avg_rec_s,
            "Train/RecA": avg_rec_a,
            "Train/KL": avg_kl,
            "Train/KL_Weight": current_kl_weight,
            "Train/LR": current_lr,
            "epoch": epoch  # 显式记录 epoch
        }

        # --- Evaluate (每 5 个 Epoch) ---
        if epoch % 5 == 0 or epoch == 1:
            eval_stats = evaluate(model, test_loader, device, scaler, state_loss_fn, lambda_state, lambda_action)

            # 判断是否 Best
            is_best = eval_stats['mpjpe'] < best_mpjpe
            if is_best:
                best_mpjpe = eval_stats['mpjpe']
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

            # ================= 打印 =================
            print("-" * 65)
            print(f"Epoch {epoch:03d}/{num_epochs} | Step {global_step} | LR {current_lr:.2e}")
            print(f"Train | Loss: {avg_loss:.4f} | KL_w: {current_kl_weight:.4f}")
            print(f"Eval  | MPJPE: {eval_stats['mpjpe']:.2f} mm {'[New Best!]' if is_best else ''}")
            print(f"      | Action: R2={eval_stats['r2_score']:.3f} | Cos={eval_stats['cosine_sim']:.3f} "
                  f"| MAE={eval_stats['mae']:.4f}")
            print(f"      | Jerk  : GT={eval_stats.get('gt_jerk', 0.0):.4f} | Pred={eval_stats.get('pred_jerk', 0.0):.4f} "
                  f"| Ratio={eval_stats.get('jerk_ratio', 0.0):.4f}")
            print(f"      | Loss  : RecS={eval_stats['rec_s']:.4f} | RecA={eval_stats['rec_a']:.4f} | KL={eval_stats['kl']:.4f}")
            print("-" * 65)
            # =================================================

            # Wandb Log (合并 Train 和 Eval)
            log_dict.update({
                "Eval/MPJPE_mm": eval_stats['mpjpe'],
                "Eval/R2": eval_stats['r2_score'],
                "Eval/CosSim": eval_stats['cosine_sim'],
                "Eval/MAE": eval_stats['mae'],
                "Eval/RecS": eval_stats['rec_s'],
                "Eval/RecA": eval_stats['rec_a'],
                "Eval/KL": eval_stats['kl'],
                "Eval/GT_Jerk": eval_stats.get('gt_jerk', 0.0),
                "Eval/Pred_Jerk": eval_stats.get('pred_jerk', 0.0),
                "Eval/Jerk_Ratio": eval_stats.get('jerk_ratio', 0.0)
            })

        else:
            # 平时只打印一行简报
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | KL_w: {current_kl_weight:.4f}")

        # --- ★★★ Wandb Log (Per Epoch) ★★★ ---
        # 关键点：设置 step=epoch，这样横坐标就是 1, 2, 3...
        wandb.log(log_dict, step=epoch)

    # Finish
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    wandb.finish()


if __name__ == "__main__":
    DATA_ROOT = "logs/smpl_ppo/amass_rollout/272_w_action"
    # DATA_ROOT = "/home/miku/Documents/VideoSkills/dataset/272_w_action/amass_train_success"
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    SAVE_DIR = f"./vae_checkpoint/less_epoch_{timestamp}"

    train_vae(
        DATA_ROOT,
        state_dim=272,  # 请确认维度
        action_dim=69,  # 请确认维度
        save_dir=SAVE_DIR,
        state_recons_type="optimal_gaussian",
        action_recons_type="mse",
        batch_size=2048,  # 大 Batch
        num_epochs=100  # 多 Epoch
    )