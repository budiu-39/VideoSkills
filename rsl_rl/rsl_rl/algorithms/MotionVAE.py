import os
import glob
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rsl_rl.network import TCNEncoder, TCNDecoder
import datetime


# =========================
# 1. VAE 模型
# =========================

class MotionVAE(nn.Module):
    """
    整体 VAE:
      输入: 状态序列 state_seq [B, T, D_state]
      输出: 重建的 state_hat, action_hat 以及 latent μ, logvar
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        window_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TCNEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.decoder = TCNDecoder(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            window_size=window_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            use_pos_encoding=True,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state_seq):
        # state_seq: [B, T, D_state]
        mu, logvar = self.encoder(state_seq)
        z = self.reparameterize(mu, logvar)
        state_hat, action_hat = self.decoder(z)
        return state_hat, action_hat, mu, logvar


# =========================
# 2. 数据集: 读取 .npy 并切 window（缓存到内存）
# =========================

class MotionWindowDataset(Dataset):
    def __init__(
        self,
        window_size: int,
        stride: int,
        action_dim: int,
        file_paths,              # <- 新增：直接传文件列表进来
        load_to_memory: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.action_dim = action_dim

        self.files = sorted(file_paths)
        if len(self.files) == 0:
            raise ValueError("No .npy files provided")

        self.data_list = []
        if load_to_memory:
            print(f"[Dataset] Loading {len(self.files)} numpy files into memory...")
            for f in self.files:
                arr = np.load(f)
                self.data_list.append(arr)
        else:
            self.data_list = None

        # index: (file_idx, start_t)
        self.index = []
        for file_idx, fpath in enumerate(self.files):
            if self.data_list is not None:
                arr = self.data_list[file_idx]
            else:
                arr = np.load(fpath)
            T = arr.shape[0]

            for start in range(0, T - window_size + 1, stride):
                self.index.append((file_idx, start))

        print(f"[Dataset] Using {len(self.files)} files, total {len(self.index)} windows.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]

        # 取数据
        if self.data_list is not None:
            arr = self.data_list[file_idx]
        else:
            arr = np.load(self.files[file_idx])

        window = arr[start:start + self.window_size]  # [W, D_total]
        D_total = window.shape[1]
        D_action = self.action_dim
        D_state = D_total - D_action

        state = window[:, :D_state]
        action = window[:, D_state:]

        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()

        return {
            "state": state,     # [W, D_state]
            "action": action,   # [W, D_action]
        }



# =========================
# 3. Loss & KL Annealing
# =========================
class OptimalGaussianReconstructionLoss(nn.Module):
    """
    Optimal-σ Gaussian NLL 重建损失:
      - mode="global": 所有维度共享一个 σ（全局 normalization）
      - mode="per_dim": 每个维度一个 σ（per-dimension normalization）
    输出使用 .mean()，数值尺度大致和 F.mse_loss 对齐。
    """
    def __init__(self, mode: str = "per_dim", min_log_sigma: float = -2.0):
        super().__init__()
        assert mode in ("global", "per_dim")
        self.mode = mode
        self.min_log_sigma = min_log_sigma

    @staticmethod
    def softclip(tensor: torch.Tensor, min_val: float) -> torch.Tensor:
        # 保证 log_sigma 不会太小
        return min_val + F.softplus(tensor - min_val)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, T, D]
        返回: 标量 loss（mean over B,T,D）
        """
        diff = target - pred  # [B, T, D]

        if self.mode == "global":
            # 所有维度共享一个 σ
            # mean over (B,T,D)
            mse_mean = diff.pow(2).mean()
            log_sigma = 0.5 * torch.log(mse_mean + 1e-8)  # sqrt -> ^0.5, 再 log
            log_sigma = self.softclip(log_sigma, self.min_log_sigma)
        else:
            # 每个 feature 维度一个 σ: mean over (B,T)，保留 D
            # 结果形状 [1,1,D] 方便广播
            mse_mean = diff.pow(2).mean(dim=(0, 1), keepdim=True)  # [1,1,D]
            log_sigma = 0.5 * torch.log(mse_mean + 1e-8)
            log_sigma = self.softclip(log_sigma, self.min_log_sigma)

        sigma_inv = torch.exp(-log_sigma)
        # 高斯 NLL：0.5 * ((x-μ)/σ)^2 + log σ + 0.5 log(2π)
        nll = 0.5 * (diff * sigma_inv) ** 2 # + log_sigma + 0.5 * np.log(2 * np.pi)
        # 用 mean，和 F.mse_loss(reduction="mean") 尺度相近
        return nll.mean()


def kl_divergence(mu, logvar):
    # KL(N(mu, sigma) || N(0, I))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def analyze_dataset_stats(dataset, batch_size=4096):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sum_state = 0.0
    sumsq_state = 0.0
    sum_action = 0.0
    sumsq_action = 0.0
    count = 0

    for batch in loader:
        state = batch["state"].float()   # [B,T,D_s]
        action = batch["action"].float() # [B,T,D_a]

        B, T, D_s = state.shape
        _, _, D_a = action.shape
        N = B * T

        state_flat = state.reshape(N, D_s)
        action_flat = action.reshape(N, D_a)

        sum_state += state_flat.sum(dim=0)
        sumsq_state += (state_flat ** 2).sum(dim=0)

        sum_action += action_flat.sum(dim=0)
        sumsq_action += (action_flat ** 2).sum(dim=0)

        count += N

    mean_state = sum_state / count           # [D_s]
    mean_action = sum_action / count         # [D_a]

    var_state = sumsq_state / count - mean_state ** 2
    var_action = sumsq_action / count - mean_action ** 2

    std_state = var_state.sqrt()
    std_action = var_action.sqrt()

    print("\n===== Dataset Stats (mean & std) =====")
    print(f"mean(|state|)  ≈ {state_flat.abs().mean().item():.6f}")
    print(f"mean(|action|) ≈ {action_flat.abs().mean().item():.6f}")
    print("--------------------------------------")
    print("Per-dim state: mean & std (first 10 dims)")
    print("mean:", mean_state[:10])
    print("std :", std_state[:10])
    print("--------------------------------------")
    print("Per-dim action: mean & std (first 10 dims)")
    print("mean:", mean_action[:10])
    print("std :", std_action[:10])

    return mean_state, std_state, mean_action, std_action


def get_kl_weight(global_step: int,
                  total_anneal_steps: int,
                  max_beta: float = 1.0) -> float:
    """
    线性 KL annealing:
      global_step = 0         -> 0
      global_step = total_anneal_steps -> max_beta
      之后保持 max_beta 不变
    """
    if total_anneal_steps <= 0:
        return max_beta
    frac = float(global_step) / float(total_anneal_steps)
    return max_beta * min(1.0, frac)

def evaluate_once(model, data_loader, device, lambda_state, lambda_action, beta_kl, kl_weight=1.0, max_batches=10,
                  state_recons_loss_fn=None):
    model.eval()
    total_loss = 0.0
    total_rec_state = 0.0
    total_rec_action = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:   # 只看前几个 batch 就够了，不用全量
                break

            state = batch["state"].to(device)
            action = batch["action"].to(device)

            state_hat, action_hat, mu, logvar = model(state)

            if state_recons_loss_fn is None:
                rec_state = F.mse_loss(state_hat, state)
            else:
                rec_state = state_recons_loss_fn(state_hat, state)

            rec_action =  F.mse_loss(action_hat, action)
            kl = kl_divergence(mu, logvar)

            loss = lambda_state * rec_state + lambda_action * rec_action + beta_kl * kl

            total_loss += loss.item()
            total_rec_state += rec_state.item()
            total_rec_action += rec_action.item()
            total_kl += kl.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_rec_s = total_rec_state / n_batches
    avg_rec_a = total_rec_action / n_batches
    avg_kl = total_kl / n_batches

    print(
        f"[Init Eval] Loss {avg_loss:.4f} | "
        f"RecS {avg_rec_s:.4f} | RecA {avg_rec_a:.4f} | KL {avg_kl:.4f} | KL_w {beta_kl:.4f}"
    )

# =========================
# 4. 训练 + 模型存储
# =========================

def train_vae(
    data_root: str,
    state_dim: int,
    action_dim: int,
    window_size: int = 32,
    stride: int = 1,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    batch_size: int = 32,
    num_epochs: int = 50,
    lr: float = 1e-3,
    lambda_state: float = 1.0,
    lambda_action: float = 2.0,
    beta_kl: float = 1.0,
    kl_anneal_steps: int = 20000,
    device: str = "cuda",
    save_dir: str = None,
    save_every: int = 5,
    state_recons_type: str = "global_nll",  # "mse", "global_nll", "per_dim_nll"
):
    """
    训练 VAE 并按 epoch 保存 checkpoint：
      - 每 save_every 个 epoch 保存一次
      - 同时保存 best_loss 模型
    """


    all_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
    n_files = len(all_files)
    n_train_files = int(n_files * 0.1)

    # 随机打乱再分
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_files)
    train_files = [all_files[i] for i in perm[:n_train_files]]
    test_files = [all_files[i] for i in perm[n_train_files:]]

    train_dataset = MotionWindowDataset(
        window_size=window_size,
        stride=stride,
        action_dim=action_dim,
        file_paths=train_files,
        load_to_memory=True,
    )

    analyze_dataset_stats(train_dataset)

    # test_dataset = MotionWindowDataset(
    #     window_size=window_size,
    #     stride=stride,
    #     action_dim=action_dim,
    #     file_paths=test_files,
    #     load_to_memory=True,
    # )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model = MotionVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    if state_recons_type == "global_nll":
        state_recons_loss_fn = OptimalGaussianReconstructionLoss(mode="global").to(device)
    elif state_recons_type == "per_dim_nll":
        state_recons_loss_fn = OptimalGaussianReconstructionLoss(mode="per_dim").to(device)
    else:
        state_recons_loss_fn = None  # 走 MSE

    print("=== Evaluate random initialization (epoch 0) ===")
    evaluate_once(
        model,
        train_loader,
        device=device,
        lambda_state=lambda_state,
        lambda_action=lambda_action,
        beta_kl=beta_kl,
        kl_weight=0.0,   # 如果你想看不带 KL 的重建误差，就设 0；想要带 KL 可以设 ~0.01
        max_batches=10,
        state_recons_loss_fn=state_recons_loss_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 模型保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rec_state = 0.0
        total_rec_action = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in train_loader:
            global_step += 1

            # kl_weight = get_kl_weight(
            #     global_step=global_step,
            #     total_anneal_steps=kl_anneal_steps,
            #     max_beta=beta_kl,
            # )

            state = batch["state"].to(device)   # [B, W, D_state]
            action = batch["action"].to(device) # [B, W, D_action]

            state_hat, action_hat, mu, logvar = model(state)

            # reconstruction losses
            if state_recons_type == "mse" or state_recons_loss_fn is None:
                rec_state = F.mse_loss(state_hat, state)

            else:
                rec_state = state_recons_loss_fn(state_hat, state)
                # rec_action = state_recons_loss_fn(action_hat, action)

            rec_action = F.mse_loss(action_hat, action)
            kl = kl_divergence(mu, logvar)

            loss = lambda_state * rec_state + lambda_action * rec_action + beta_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec_state += rec_state.item()
            total_rec_action += rec_action.item()
            total_kl += kl.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_rec_s = total_rec_state / n_batches
        avg_rec_a = total_rec_action / n_batches
        avg_kl = total_kl / n_batches

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {avg_loss:.4f} | "
            f"RecS {avg_rec_s:.4f} | "
            f"RecA {avg_rec_a:.4f} | "
            f"KL {avg_kl:.4f} | "
            # f"KL_w {kl_weight:.4f}"
        )

        # ====== 模型保存 ======
        if save_dir is not None:
            # 1) 每 save_every 个 epoch 保存一个 checkpoint
            if (epoch % save_every) == 0:
                ckpt_path = os.path.join(save_dir, f"motion_vae_epoch{epoch:03d}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_loss": avg_loss,
                        "config": {
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                            "latent_dim": latent_dim,
                            "window_size": window_size,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "kernel_size": kernel_size,
                            "dropout": dropout,
                        },
                    },
                    ckpt_path,
                )
                print(f"[Save] Checkpoint saved to {ckpt_path}")

            # 2) 记录 best loss 模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "motion_vae_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_loss": avg_loss,
                        "config": {
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                            "latent_dim": latent_dim,
                            "window_size": window_size,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "kernel_size": kernel_size,
                            "dropout": dropout,
                        },
                    },
                    best_path,
                )
                print(f"[Save] Best model updated at epoch {epoch}, loss={best_loss:.4f}")

    return model


if __name__ == "__main__":
    # 使用示例
    DATA_ROOT = "/home/miku/Documents/VideoSkills/logs/smpl_ppo/amass_rollout/296_w_action"
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    save_dir = f"./vae_checkpoint/run_{timestamp}"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = train_vae(
        data_root=DATA_ROOT,
        state_dim=296,
        action_dim=69,
        window_size=32,
        stride=2,          # 建议用 1 或 2，看数据量
        latent_dim=64,
        hidden_dim=256,
        num_layers=4,
        kernel_size=3,
        dropout=0.1,
        batch_size=2048,
        num_epochs=200,
        lr=1e-3,
        lambda_state=1.0,
        lambda_action=10.0,     # action 更重要一点
        beta_kl=0.01,
        kl_anneal_steps=15000, # KL annealing 的总 step 数
        device=device,
        save_dir=save_dir,  # 模型保存目录
        save_every=5,          # 每 5 个 epoch 存一个 checkpoint
    )
