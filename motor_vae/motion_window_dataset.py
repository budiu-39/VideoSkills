import numpy as np
import torch
from torch.utils.data import Dataset
import os


class MotionWindowDataset(Dataset):
    def __init__(self, window_size, stride, action_dim, file_paths, load_to_memory=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.action_dim = action_dim
        self.files = sorted(file_paths)
        self.data_list = []
        self.load_to_memory = load_to_memory

        if self.load_to_memory:
            print(f"[Dataset] Loading {len(self.files)} files...")
            for f in self.files:
                self.data_list.append(np.load(f))

        self.index = []
        for file_idx, fpath in enumerate(self.files):
            # 获取数据长度
            if self.load_to_memory:
                T = self.data_list[file_idx].shape[0]
            else:
                # 为了速度，不读取整个文件只看 shape，但这需要加载一下头信息
                # 这里简化处理，假设预先知道或快速读取
                arr = np.load(fpath, mmap_mode='r')
                T = arr.shape[0]

            # 生成窗口索引
            for start in range(0, T - window_size + 1, stride):
                self.index.append((file_idx, start))

        print(f"[Dataset] Total windows: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]

        if self.load_to_memory:
            arr = self.data_list[file_idx]
        else:
            arr = np.load(self.files[file_idx])  # 或者 mmap

        window = arr[start:start + self.window_size]  # [Window, D_total]

        # ★★★ 核心修改：自动检测是否包含 Action ★★★
        D_total = window.shape[1]

        # 假设 State 总是位于前 272 维
        state_dim = D_total - self.action_dim

        if D_total == 272 + 69:
            # 情况 A: 标准训练数据 [State | Action]
            state = window[:, :state_dim]
            action = window[:, state_dim:]
        elif D_total == 272:
            # 情况 B: In-the-wild 数据 [State] (无 Action)
            state = window
            # 生成假的 Action (全0)，保证 DataLoader 不报错
            action = np.zeros((self.window_size, self.action_dim), dtype=state.dtype)

        return {
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(action).float()
        }


class TargetedMotionDataset(MotionWindowDataset):
    """
    继承自 MotionWindowDataset。
    复用其数据加载和预处理逻辑，但覆盖索引生成逻辑。
    只针对 target_frames 列表中的 (文件, 帧数) 生成以该帧为中心的窗口。
    """

    def __init__(self, target_frames, window_size, action_dim, load_to_memory=True):
        # 1. 提取所有涉及到的文件路径 (去重)
        # target_frames结构: [(file_path, frame_idx), ...]
        unique_files = sorted(list(set([t[0] for t in target_frames])))

        # 2. 调用父类初始化
        # 这会自动加载 unique_files 中的数据到 self.data_list
        # 并生成默认的滑动窗口索引到 self.index (我们稍后会覆盖它)
        super().__init__(
            window_size=window_size,
            stride=1,  # 这里stride多少无所谓，反正会被覆盖
            action_dim=action_dim,
            file_paths=unique_files,
            load_to_memory=load_to_memory
        )

        # 3. ★★★ 覆盖 self.index ★★★
        # 建立 文件路径 -> data_list索引 的映射
        file_to_idx = {f: i for i, f in enumerate(self.files)}

        self.index = []  # 清空父类生成的滑动窗口索引

        print(f"[TargetedDataset] Re-indexing for {len(target_frames)} specific frames...")

        for fpath, center_frame in target_frames:
            if fpath not in file_to_idx:
                continue

            f_idx = file_to_idx[fpath]

            # 获取该文件的总长度
            # 注意：父类已经把数据加载到 self.data_list (如果 load_to_memory=True)
            if self.data_list:
                T = self.data_list[f_idx].shape[0]
            else:
                # 如果没加载到内存，只能临时读一下 shape (为了性能建议 load_to_memory=True)
                # 这里简单处理，假设一定在内存
                arr = np.load(fpath, mmap_mode='r')
                T = arr.shape[0]

            # 计算窗口起始位置：让 center_frame 居中
            half_win = window_size // 2
            start = center_frame - half_win

            # 边界 Clamp：防止越界
            # 如果窗口超出左边，取 0；如果超出右边，取 T - window_size
            start = max(0, min(start, T - window_size))

            # 只有当文件长度足够时才添加
            if T >= window_size:
                # 父类的 __getitem__ 需要 (file_idx, start)
                self.index.append((f_idx, start))

        print(f"[TargetedDataset] Final valid windows: {len(self.index)}")