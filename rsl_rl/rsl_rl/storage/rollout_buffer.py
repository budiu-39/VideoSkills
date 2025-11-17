import torch

class ReplayBuf:
    """
    聚合缓冲（DAgger）：数据常驻 CPU，随机抽样时才搬到 GPU。
    避免每次训练把所有块 cat 到一个超大 GPU 张量导致 OOM。
    """
    def __init__(self, capacity, device, dtype=torch.float32, pin_memory=True):
        self.device = device
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.dtype = dtype

        self.obs_chunks = []
        self.mu_chunks  = []
        self.std_chunks = []
        self.val_chunks = []
        self.size = 0  # 样本总数

    def _to_cpu(self, x):
        x = x.detach().to('cpu', dtype=self.dtype, non_blocking=False)
        if self.pin_memory:
            try:
                x = x.pin_memory()
            except:
                pass
        return x

    def add(self, obs, mu, std, val):
        # 所有新数据搬到 CPU 存（避免 GPU 长驻）
        self.obs_chunks.append(self._to_cpu(obs))
        self.mu_chunks.append(self._to_cpu(mu))
        self.std_chunks.append(self._to_cpu(std))
        self.val_chunks.append(self._to_cpu(val))
        self.size += obs.shape[0]

        # 截断容量：从最早的块开始删
        while self.size > self.capacity and len(self.obs_chunks) > 0:
            popped_n = self.obs_chunks[0].shape[0]
            self.obs_chunks.pop(0); self.mu_chunks.pop(0); self.std_chunks.pop(0); self.val_chunks.pop(0)
            self.size -= popped_n

    def __len__(self):
        return self.size

    def sample_batches(self, batch_size):
        """
        随机抽样（按 chunk 向量化）：
          1) 在全局长度上洗牌得到 idx_global
          2) 用 np.searchsorted 一次性映射到 (chunk_id, offset)
          3) 对每个命中的 chunk 批量 index_select
          4) 每个 chunk 只搬一次到 GPU，最后 cat
        """
        if self.size == 0:
            return

        import numpy as np

        lens = np.fromiter((t.shape[0] for t in self.obs_chunks), dtype=np.int64)
        cumsum = np.concatenate(([0], np.cumsum(lens)))  # len = n_chunks+1
        N = int(cumsum[-1])

        idx_global = torch.randperm(N)  # CPU
        for s in range(0, N, batch_size):
            idx = idx_global[s: s + batch_size].numpy()

            # 一次性映射到 chunk 与 offset
            chunk_ids = np.searchsorted(cumsum, idx, side='right') - 1
            offs = idx - cumsum[chunk_ids]

            # 按 chunk 聚合，批量索引
            uniq = np.unique(chunk_ids)
            obs_mb_list, mu_mb_list, std_mb_list, val_mb_list = [], [], [], []
            for cid in uniq:
                sel = np.where(chunk_ids == cid)[0]
                sel_offs = torch.as_tensor(offs[sel], dtype=torch.long)

                # 在 CPU 上一次性 gather；随后一次性搬到 GPU
                obs_mb_list.append(self.obs_chunks[cid].index_select(0, sel_offs))
                mu_mb_list.append(self.mu_chunks[cid].index_select(0, sel_offs))
                std_mb_list.append(self.std_chunks[cid].index_select(0, sel_offs))
                val_mb_list.append(self.val_chunks[cid].index_select(0, sel_offs))

            obs_mb = torch.cat(obs_mb_list, dim=0).to(self.device, non_blocking=True)
            mu_mb = torch.cat(mu_mb_list, dim=0).to(self.device, non_blocking=True)
            std_mb = torch.cat(std_mb_list, dim=0).to(self.device, non_blocking=True)
            val_mb = torch.cat(val_mb_list, dim=0).to(self.device, non_blocking=True)
            yield obs_mb, mu_mb, std_mb, val_mb
