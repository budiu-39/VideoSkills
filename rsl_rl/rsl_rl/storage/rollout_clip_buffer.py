import torch
import random

class SegmentReplayBuf:
    """
    以 [T, obs_dim] 的环境片段为最小存储单元。
    每次 rollout 会产生 B 个 segment（每个 env 一个）。
    训练时一次随机抽 K 个 segment -> 拼成 [T*K, obs_dim] 做一个 batch。

    版本说明：
    - 所有数据都常驻在 self.device 上（GPU）
    - 不再进行 CPU/GPU 来回拷贝
    """
    def __init__(self, capacity_segments, steps_per_env, device,
                 dtype=torch.float32, pin_memory=True):
        self.capacity = capacity_segments    # 最多存多少个 segment，而不是 frame 数
        self.T = steps_per_env
        self.device = device
        self.dtype = dtype
        self.pin_memory = pin_memory  # 已经没用，但保留这个参数避免其他代码出错

        self.obs_segs   = []  # 每个元素 shape = [T, obs_dim] (on device)
        self.mu_segs    = []  # [T, act_dim]
        self.std_segs   = []  # [T, act_dim]
        self.val_segs   = []  # [T]
        self.reset_segs = []  # [T]  (step_t -> step_{t+1} 的 reset 标志)

    def _to_device(self, x: torch.Tensor):
        """所有存入 buffer 的 tensor 都：
        - detach 掉梯度
        - 转到指定 device
        - cast 到指定 dtype（除了 reset 这种 bool/long，可以自己控制）
        """
        x = x.detach()
        # reset_flat 通常是 bool/uint8，这里如果你想保持类型，可以在调用处跳过 dtype cast
        if x.dtype.is_floating_point:
            x = x.to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            x = x.to(self.device, non_blocking=True)
        return x

    def add_rollout(self, obs_flat, mu_flat, std_flat, val_flat, reset_flat, num_envs):
        """
        将一个 [T*B, ...] 的 rollout 拆成 B 个 [T, ...] 的 segment 逐个存入 buffer
        这里假设 obs_flat 等可以在 CPU 或 GPU 上，统一搬到 self.device。
        """
        T = self.T
        B = num_envs

        obs_flat   = self._to_device(obs_flat)
        mu_flat    = self._to_device(mu_flat)
        std_flat   = self._to_device(std_flat)
        val_flat   = self._to_device(val_flat)
        reset_flat = self._to_device(reset_flat)

        # 还原成 [T, B, ...]
        obs_seq   = obs_flat.view(T, B, -1)
        mu_seq    = mu_flat.view(T, B, -1)
        std_seq   = std_flat.view(T, B, -1)
        val_seq   = val_flat.view(T, B)
        reset_seq = reset_flat.view(T, B)

        for b in range(B):
            # 这里的 tensor 已经在 device 上了
            self.obs_segs.append(obs_seq[:, b].contiguous())    # [T, obs_dim]
            self.mu_segs.append(mu_seq[:, b].contiguous())
            self.std_segs.append(std_seq[:, b].contiguous())
            self.val_segs.append(val_seq[:, b].contiguous())
            self.reset_segs.append(reset_seq[:, b].contiguous())

        # FIFO 截断（按 segment 数目截）
        while len(self.obs_segs) > self.capacity:
            self.obs_segs.pop(0)
            self.mu_segs.pop(0)
            self.std_segs.pop(0)
            self.val_segs.pop(0)
            self.reset_segs.pop(0)

    def __len__(self):
        return len(self.obs_segs)

    def sample_segments(self, num_segments):
        """
        随机抽 num_segments 个 segment，返回拼接好的 flat batch:
          obs_mb   : [num_segments*T, obs_dim]
          mu_mb    : [num_segments*T, act_dim]
          std_mb   : ...
          val_mb   : ...
          reset_mb : [num_segments*T]

        所有返回值都在 self.device 上。
        """
        assert len(self.obs_segs) > 0, "Buffer is empty"
        num_segments = min(num_segments, len(self.obs_segs))
        idxs = random.sample(range(len(self.obs_segs)), num_segments)

        obs_list, mu_list, std_list, val_list, reset_list = [], [], [], [], []
        for i in idxs:
            obs_list.append(self.obs_segs[i])    # 已经在 device 上
            mu_list.append(self.mu_segs[i])
            std_list.append(self.std_segs[i])
            val_list.append(self.val_segs[i])
            reset_list.append(self.reset_segs[i])

        # [num_segments, T, ...] -> [T, num_segments, ...] -> [T*num_segments, ...]
        # 因为都已经在 device 上，不需要再 .to(self.device)
        obs_mb   = torch.stack(obs_list, dim=1)       # [T, K, obs_dim]
        mu_mb    = torch.stack(mu_list,  dim=1)
        std_mb   = torch.stack(std_list, dim=1)
        val_mb   = torch.stack(val_list, dim=1)
        reset_mb = torch.stack(reset_list,dim=1)

        T = self.T
        K = num_segments
        obs_mb   = obs_mb.view(T*K, -1)
        mu_mb    = mu_mb.view(T*K, -1)
        std_mb   = std_mb.view(T*K, -1)
        val_mb   = val_mb.view(T*K)
        reset_mb = reset_mb.view(T*K)

        return obs_mb, mu_mb, std_mb, val_mb, reset_mb

    def iter_segment_batches(self, segments_per_batch, shuffle=True):
        import numpy as np
        if len(self.obs_segs) == 0:
            return

        n_seg = len(self.obs_segs)
        idxs = np.arange(n_seg)
        if shuffle:
            np.random.shuffle(idxs)

        T = self.T

        # 一次性 stack 所有 segment，后面 batch 里只做索引
        # 形状: [T, n_seg, ...]
        obs_all = torch.stack(self.obs_segs, dim=1)  # [T, n_seg, obs_dim]
        mu_all = torch.stack(self.mu_segs, dim=1)  # [T, n_seg, act_dim]
        std_all = torch.stack(self.std_segs, dim=1)  # [T, n_seg, act_dim]
        val_all = torch.stack(self.val_segs, dim=1)  # [T, n_seg]
        reset_all = torch.stack(self.reset_segs, dim=1)  # [T, n_seg]

        for s in range(0, n_seg, segments_per_batch):
            batch_ids = idxs[s:s + segments_per_batch]
            K = len(batch_ids)
            if K == 0:
                continue

            # 结果: [T, K, ...]
            obs_mb = obs_all[:, batch_ids]  # [T, K, obs_dim]
            mu_mb = mu_all[:, batch_ids]
            std_mb = std_all[:, batch_ids]
            val_mb = val_all[:, batch_ids]
            reset_mb = reset_all[:, batch_ids]

            # 展平成 time-major: [T*K, ...]
            obs_mb = obs_mb.reshape(T * K, -1)
            mu_mb = mu_mb.reshape(T * K, -1)
            std_mb = std_mb.reshape(T * K, -1)
            val_mb = val_mb.reshape(T * K)
            reset_mb = reset_mb.reshape(T * K)

            yield obs_mb, mu_mb, std_mb, val_mb, reset_mb

