import os
import glob
import torch
import numpy as np
import argparse
from tqdm import tqdm

# 引入你的 VAE 接口
from motor_vae.vae_wrapper import VAEInterface
from scripts.rep_272.plot_3d_global import draw_to_batch
from scripts.rep_272.recover_visualize import recover_from_272_zup


def process_single_file(vae, file_path, output_dir, stride=4):
    """
    处理单个 272D 动作文件。
    1. 以 Stride=1 密集生成全序列的 Z 和 Recon。
    2. 按 Stride=4 切分 Z，但保存完整的 Recon。
    """
    filename = os.path.basename(file_path).replace('.npy', '')

    # 1. 加载数据
    try:
        raw_motion = np.load(file_path)  # [T, 272]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # 转换为 Tensor 并移动到设备
    motion_tensor = torch.tensor(raw_motion, dtype=torch.float32, device=vae.device)

    L, D = motion_tensor.shape
    window_size = vae.window_size

    # 如果动作太短，无法形成一个窗口，跳过
    if L < window_size:
        return

    # 2. 创建密集滑动窗口 (Stride=1) 用于生成连续的 Recon
    # [N_dense, Window, D] where N_dense = L - Window + 1
    all_windows = motion_tensor.unfold(0, window_size, 1).permute(0, 2, 1)

    # 3. 批量处理 (Batch Processing)
    # 为了防止显存爆炸，我们需要分批处理这些密集的窗口
    batch_size = 256
    z_dense_list = []
    recon_dense_list = []

    num_windows = all_windows.shape[0]

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = all_windows[i: i + batch_size]  # [B, Win, D]

            # Encode -> Z
            mu = vae.encode_motion(batch)  # [B, Latent]
            z_dense_list.append(mu)

            # Decode -> Recon Window
            recon_win = vae.decode_latent(mu)  # [B, Win, D]

            # [关键]: 为了得到连续的 motion_length 序列，我们取每个窗口的第一帧
            # 这样拼起来就是连续的 [t, t+1, t+2...]
            # (除了最后几帧，但通常 RL 训练不关心最后那个尾巴)
            recon_frame = recon_win[:, 0, :]  # [B, D]
            recon_dense_list.append(recon_frame)

    # 4. 拼接得到密集序列
    z_dense = torch.cat(z_dense_list, dim=0)  # [N_dense, Latent]
    recon_dense = torch.cat(recon_dense_list, dim=0)  # [N_dense, 272]

    # 转回 CPU numpy
    z_dense_np = z_dense.cpu().numpy()
    recon_dense_np = recon_dense.cpu().numpy()

    # 5. 按照 stride=4 (0, 1, 2, 3) 进行分组保存
    # Z 进行切片 (1:4)，Recon 保持完整 (1:1)
    for phase in range(stride):
        # 切片 Z: 从 phase 开始，每隔 4 帧取一个
        z_subset = z_dense_np[phase::stride]
        recon_subset = recon_dense_np[phase:]
        # 边界检查：如果切片后为空，跳过
        if len(z_subset) == 0:
            continue

        if phase > 0:  # TODO: 这里先只生成 sub 1 试试？
            continue

        # 构造输出文件名
        save_name = f"{filename}_sub{phase}.npy"
        save_path = os.path.join(output_dir, save_name)

        data_dict = {
            "z": z_subset,  # [Length/4, Latent_Dim] -> 稀疏控制信号
            "recon": recon_subset  # [Length, 272]          -> 完整物理参考轨迹
        }

        np.save(save_path, data_dict)

        visualize = False
        if visualize:
            pred_xyz = recover_from_272_zup(recon_dense_np, 22)
            # gt_xyz = motion_norm_22.global_translation.clone().float().numpy()
            # err_l, err_g = calc_mpjpe(gt_xyz, pred_xyz)
            # if err_g > 1.0: print(f"Warning: Large error in {src_path}")

            file_name = os.path.basename(file_path).replace('.npy', '')
            output_dir = 'rollout'
            out_path_rec = os.path.join(output_dir, f"{file_name}_recovered.mp4")
            draw_to_batch(
                pred_xyz.reshape(1, -1, 22, 3),
                outname=[out_path_rec],
                fps=30,
                kinetic_chain='sim_22'
            )
            print(f"Saved: {out_path_rec}")





def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================
    # 路径自动查找逻辑
    # ==========================
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Directory not found: {args.model_dir}")

    try:
        config_path = glob.glob(os.path.join(args.model_dir, "*.yaml"))[0]
    except IndexError:
        raise FileNotFoundError(f"No .yaml file found in {args.model_dir}")

    pt_files = glob.glob(os.path.join(args.model_dir, "*.pt"))
    scaler_path = None
    ckpt_path = None

    for f in pt_files:
        name = os.path.basename(f)
        if "stat" in name or "scaler" in name:
            scaler_path = f
        else:
            ckpt_path = f

    if not scaler_path or not ckpt_path:
        raise FileNotFoundError("Check model dir files.")

    print(f"[Auto-Load] Config: {os.path.basename(config_path)}, Model: {os.path.basename(ckpt_path)}")

    # ==========================
    # 初始化
    # ==========================
    vae = VAEInterface(
        config_path=config_path,
        model_ckpt_path=ckpt_path,
        scaler_stats_path=scaler_path,
        device=device
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取文件
    file_pattern = os.path.join(args.input_dir, "**", "*.npy")
    all_files = glob.glob(file_pattern, recursive=True)
    all_files = [f for f in all_files if "_global" not in f and "_sub" not in f]

    print(f"Found {len(all_files)} files to process.")

    for fpath in tqdm(all_files):
        process_single_file(vae, fpath, args.output_dir, stride=4)

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing .yaml, model.pt, and stats.pt")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing 272D .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .npy files")
    args = parser.parse_args()
    main(args)