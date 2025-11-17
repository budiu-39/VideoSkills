import torch

class WindowReplayBuffer:
    def __init__(self, T, max_size):
        self.T = T
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.s_phys = None   # [N, T, D_phys]
        self.s_g    = None   # [N, T, D_ref]
        self.a_teach = None  # [N, T, D_act]
        self.motion_key = []
        self.win_idx    = []

    def add(self, s_phys, s_g, a_teach, motion_key, win_idx):
        # s_phys, s_g, a_teach: [T, ...]
        if self.s_phys is None:
            N = self.max_size
            T = self.T
            self.s_phys  = torch.zeros((N, T, s_phys.shape[-1]), dtype=torch.float32)
            self.s_g     = torch.zeros((N, T, s_g.shape[-1]), dtype=torch.float32)
            self.a_teach = torch.zeros((N, T, a_teach.shape[-1]), dtype=torch.float32)

        self.s_phys[self.ptr].copy_(s_phys)
        self.s_g[self.ptr].copy_(s_g)
        self.a_teach[self.ptr].copy_(a_teach)

        if len(self.motion_key) < self.max_size:
            self.motion_key.append(motion_key)
            self.win_idx.append(win_idx)
        else:
            self.motion_key[self.ptr] = motion_key
            self.win_idx[self.ptr] = win_idx

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batches(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        yield (
            self.s_phys[idx],      # [B,T,D_phys]
            self.s_g[idx],         # [B,T,D_ref]
            self.a_teach[idx],     # [B,T,D_act]
            idx,                   # [B] → 用来 index meta if needed
        )
