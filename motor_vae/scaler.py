import torch

class StandardScaler:
    def __init__(self, mean=None, std=None, device="cpu", epsilon=1e-6):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def fit(self, data_loader):
        print("[Scaler] Computing mean and std...")
        sum_state = 0
        sum_sq_state = 0
        count = 0

        # 只计算 State 的 mean/std (Action 一般也可以归一化，视情况而定)
        # 这里假设对 state 和 action 都做归一化
        sum_action = 0
        sum_sq_action = 0

        for batch in data_loader:
            state = batch["state"]
            action = batch["action"]

            # Flatten B and T
            state = state.view(-1, state.shape[-1])
            action = action.view(-1, action.shape[-1])

            sum_state += state.sum(dim=0)
            sum_sq_state += (state ** 2).sum(dim=0)

            sum_action += action.sum(dim=0)
            sum_sq_action += (action ** 2).sum(dim=0)

            count += state.shape[0]

        self.mean_state = sum_state / count
        self.std_state = torch.sqrt((sum_sq_state / count) - self.mean_state ** 2 + self.epsilon)

        self.mean_action = sum_action / count
        self.std_action = torch.sqrt((sum_sq_action / count) - self.mean_action ** 2 + self.epsilon)

        print("[Scaler] Done.")
        return self

    def to(self, device):
        self.device = device
        if hasattr(self, 'mean_state'):
            self.mean_state = self.mean_state.to(device)
            self.std_state = self.std_state.to(device)
            self.mean_action = self.mean_action.to(device)
            self.std_action = self.std_action.to(device)
        return self

    def transform_state(self, state):
        return (state - self.mean_state) / self.std_state

    def inverse_transform_state(self, state_norm):
        return state_norm * self.std_state + self.mean_state

    def transform_action(self, action):
        return (action - self.mean_action) / self.std_action

    def inverse_transform_action(self, action_norm):
        return action_norm * self.std_action + self.mean_action
