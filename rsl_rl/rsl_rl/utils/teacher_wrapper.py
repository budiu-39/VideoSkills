# 在 distill 脚本顶部额外引入
from rsl_rl.utils.running_mean_std import RunningMeanStd
import torch
import torch.nn as nn

class TeacherWrapper(nn.Module):
    def __init__(self, teacher_ac, ckpt_dict, num_obs, device):
        super().__init__()
        self.ac = teacher_ac
        self.device = device

        # 1) 加载老师的 obs RMS（名字在不同保存点可能不同）
        self.obs_rms = None
        for k in ["obs_rms_state_dict", "obs_mean_std_state_dict", "obs_mean_std"]:
            if k in ckpt_dict and ckpt_dict[k] is not None:
                self.obs_rms = RunningMeanStd((num_obs,)).to(device)
                self.obs_rms.load_state_dict(ckpt_dict[k])
                self.obs_rms.eval()
                break

        # 冻结版（模仿 PPONorm._refresh_temp_rms 后用于推理）
        self.obs_rms_frozen = None
        if self.obs_rms is not None:
            import copy
            self.obs_rms_frozen = copy.deepcopy(self.obs_rms)
            self.obs_rms_frozen.freeze()

    @torch.no_grad()
    def _norm_obs(self, obs):
        if self.obs_rms_frozen is None:
            return obs
        return self.obs_rms_frozen(obs)

    @torch.no_grad()
    def act_like_training(self, obs):
        """
        复刻 PPO 采样阶段：PPONorm.act() 内部对 actor_critic.act(obs) 的调用。
        这里返回的动作=老师在环境里“真的会执行”的动作（包含模型里 tanh/clip 等后处理）。
        """
        obs_n = self._norm_obs(obs)
        return self.ac.act(obs_n)   # 这条路径就是 PPONorm.act 里使用的那条 :contentReference[oaicite:2]{index=2}

    @torch.no_grad()
    def dist_params(self, obs):
        """
        用于 KL 蒸馏：返回老师的 μ/σ（分布空间，未经过 tanh/clip 的那套）。
        """
        obs_n = self._norm_obs(obs)
        self.ac.update_distribution(obs_n)
        return self.ac.action_mean, self.ac.action_std
