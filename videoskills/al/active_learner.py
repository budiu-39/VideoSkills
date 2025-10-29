# active_learning_mask.py
import torch
from typing import Literal
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA

class ActiveLearner:
    """
    轻量级 Active Learning 管理器
    - 输入为 dict: {"key_list": [...], "embedding": torch.Tensor [N, D]}
    - 内部使用 bool mask 表示 success / failed
    - reference 直接为 torch.Tensor
    """

    def __init__(self, samples: dict, reference: dict):
        self.keys = list(samples.keys())
        self.emb = torch.stack([
            torch.as_tensor(v, dtype=torch.float32)
            for v in samples.values()
        ])
        self.N = len(self.keys)
        self._key_to_idx = {k: i for i, k in enumerate(self.keys)}

        # 状态 mask
        device = self.emb.device
        self.success_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
        self.failed_mask  = torch.zeros(self.N, dtype=torch.bool, device=device)

        # reference
        self.reference = torch.stack([
            torch.as_tensor(v, dtype=torch.float32)
            for v in reference.values()
        ])
        self.seed = 42

    def keys_to_idx(self, keys):
        return torch.tensor([self._key_to_idx[k] for k in keys], dtype=torch.long, device=self.emb.device)

    # ---------- 更新 ----------
    def add_results(
        self,
        success_keys,
        failed_keys = None,
        add_success_to_ref: bool = True,
    ):
        # Success
        if success_keys:
            s_idx = self.keys_to_idx(success_keys)
            self.success_mask[s_idx] = True
            self.failed_mask[s_idx]  = False
            if add_success_to_ref:
                add = self.emb[s_idx]
                self.reference = add if self.reference is None else torch.cat([self.reference, add], dim=0)
        # Failed
        if failed_keys:
            f_idx = self.keys_to_idx(failed_keys)
            self.failed_mask[f_idx]  = True
            self.success_mask[f_idx] = False

    # ---------- 查询 ----------
    @property
    def unseen_mask(self):
        return ~(self.success_mask | self.failed_mask)

    def get_indices(self, mode: Literal["failed", "unseen", "failed_or_unseen"] = "failed_or_unseen"):
        if mode == "failed":
            return self.failed_mask.nonzero(as_tuple=True)[0]
        elif mode == "unseen":
            return self.unseen_mask.nonzero(as_tuple=True)[0]
        elif mode == "success":
            return self.success_mask.nonzero(as_tuple=True)[0]
        else:
            return (self.failed_mask | self.unseen_mask).nonzero(as_tuple=True)[0]

    def random_select(self, mode="failed_or_unseen", n_select: int = 1):
        """
        随机选择 n_select 个样本，返回 (selected_keys, selected_indices)
        """
        q_idx = self.get_indices(mode)
        n_available = len(q_idx)
        if n_available == 0:
            return [], q_idx[:0]

        n_select_eff = min(n_select, n_available)
        rng = np.random.default_rng(seed=self.seed)
        selected_local = rng.choice(n_available, size=n_select_eff, replace=False)
        selected_indices = q_idx[torch.as_tensor(selected_local, dtype=torch.long, device=q_idx.device)]
        selected_keys = [self.keys[i.item()] for i in selected_indices]

        return selected_keys, selected_indices

    # ---------- 构造 KNN 输入 ----------
    def build_knn_inputs(self, mode="failed_or_unseen", pca_reduce_to: int = None):
        """
        返回 (X_query_np, X_ref_np, q_idx)
        - X_ref = self.reference
        - X_query = 当前 mode 对应的样本 embedding
        统一转为 numpy，便于 sklearn 使用。
        """
        if self.reference is None or self.reference.numel() == 0:
            raise ValueError("reference 为空")
        q_idx = self.get_indices(mode)
        X_query = self.emb[q_idx]

        # 默认不降维；需要时可打开 PCA 到低维（更稳健）
        if pca_reduce_to is not None and pca_reduce_to < self.reference.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_reduce_to)
            X_ref_np = pca.fit_transform(self.reference.detach().cpu().numpy())
            X_query_np = pca.transform(X_query.detach().cpu().numpy())
        else:
            X_ref_np = self.reference.detach().cpu().numpy()
            X_query_np = X_query.detach().cpu().numpy()

        return X_query_np, X_ref_np, q_idx

    @staticmethod
    def _knn_density_np(X_query: np.ndarray, X_ref: np.ndarray, k: int = 15, eps: float = 1e-8) -> np.ndarray:
        if X_ref.shape[0] == 0:
            raise ValueError("X_ref is empty")
        k_eff = min(k, X_ref.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1)
        nbrs.fit(X_ref)
        dists, _ = nbrs.kneighbors(X_query, return_distance=True)
        avg_dist = dists.mean(axis=1) if dists.size > 0 else np.array([])
        return 1.0 / (avg_dist + eps)

    def select_by_reference_density(self, mode="failed_or_unseen", k=15, quantile=0.8, n_select: int = None):
        """
        基于 reference 的密度选择样本
        返回 (selected_keys, selected_indices, densities)
        - densities: 越大表示越密集，越不值得选择
        """

        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        X_query_np, X_ref_np, q_idx = self.build_knn_inputs(mode)
        if X_query_np.shape[0] == 0:
            return [], q_idx[:0], np.array([])

        # 统一的密度：query 相对 reference 的 KNN 平均距离倒数
        k_eff = max(1, min(k, X_ref_np.shape[0]))
        nbrs = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(X_ref_np)
        dists, _ = nbrs.kneighbors(X_query_np, return_distance=True)
        densities = 1.0 / (dists.mean(axis=1) + 1e-8)

        # 先按分位阈值得到“初选”
        thr = np.quantile(densities, quantile)
        selected_mask = densities >= thr  # 稀疏改法：densities <= np.quantile(densities, 1-quantile)
        selected_idx = np.where(selected_mask)[0]

        # 如果没要求固定数量，直接返回
        if n_select is None or  len(selected_idx) == n_select:
            selected_indices = q_idx[torch.as_tensor(selected_idx, dtype=torch.long, device=q_idx.device)]
            selected_keys = [self.keys[i.item()] for i in selected_indices]
            return selected_keys, selected_indices, densities

        n_select = max(0, int(n_select))
        N = len(densities)
        rng = np.random.default_rng(seed=self.seed)

        if len(selected_idx) > n_select:
            # 从已选里随机下采样到 n_select
            selected_idx = rng.choice(selected_idx, size=n_select, replace=False)

        elif len(selected_idx) < n_select:
            need = n_select - len(selected_idx)

            mask = np.ones(len(q_idx), dtype=bool)
            mask[selected_idx] = False
            remain_local = np.nonzero(mask)[0]
            if len(remain_local) < need:
                need = len(remain_local)
            chosen_extra_local = rng.choice(remain_local, size=need, replace=False)

            selected_idx = np.concatenate([selected_idx, chosen_extra_local])


        selected_indices = q_idx[torch.as_tensor(selected_idx, dtype=torch.long, device=q_idx.device)]
        selected_keys = [self.keys[i.item()] for i in selected_indices]

        return selected_keys, selected_indices, densities


    # ---------- 简单统计 ----------
    def summary(self):
        return {
            "total": self.N,
            "success": int(self.success_mask.sum()),
            "failed": int(self.failed_mask.sum()),
            "unseen": int(self.unseen_mask.sum()),
            "reference": 0 if self.reference is None else len(self.reference),
        }
