# --- videoskills/utils/convergence_monitor.py (instrumented for wandb diagnostics) ---
import numpy as np
from collections import deque

class ConvergenceMonitor:
    def __init__(self, N=300, alpha=1, cv_thr=0.08, trend_scale=1e-3,
                 patience=3, target_success=0.5, success_plateau_eps=0.01, success_plateau_k=3):
        self.N = N
        self.alpha = alpha
        self.cv_thr = cv_thr
        self.trend_scale = trend_scale
        self.patience = patience
        self.target_success = target_success
        self.success_plateau_eps = success_plateau_eps
        self.success_plateau_k = success_plateau_k
        self.pat_cnt = 0
        self.success_hist = deque(maxlen=20)
        # diagnostics from the last update(), exposed for wandb
        self.last_diag = {}

    def _ewma(self, x):
        m = x[0]; out=[]
        for r in x:
            m = self.alpha * r + (1 - self.alpha) * m
            out.append(m)
        return np.asarray(out)

    def _trend_ok(self, ewma):
        t = np.arange(len(ewma)); t_mean = t.mean()
        slope = np.dot(t - t_mean, ewma - ewma.mean()) / (np.dot(t - t_mean, t - t_mean) + 1e-8)
        p5, p95 = np.percentile(ewma, [5, 95])
        eps_trend = self.trend_scale * max(p95 - p5, 1.0)
        return abs(slope) < eps_trend, float(slope), float(p95 - p5), float(eps_trend)

    def _variance_ok(self, ewma):
        mean = float(np.mean(ewma)); std = float(np.std(ewma))
        if abs(mean) > 1e-3:
            cv = std / (abs(mean) + 1e-6)
            return (cv < self.cv_thr), mean, std, float(cv)
        else:
            iqr = np.percentile(ewma, 75) - np.percentile(ewma, 25)
            ratio = std / (iqr + 1e-6)
            # use a surrogate cv diagnostic when mean≈0
            return (ratio < 0.5), mean, std, float(ratio)

    def _quality_ok(self):
        if self.target_success is None or len(self.success_hist) == 0:
            return True
        return self.success_hist[-1] >= self.target_success

    def _success_plateau(self):
        if len(self.success_hist) < self.success_plateau_k + 1:
            return False
        recent = list(self.success_hist)[-(self.success_plateau_k+1):]
        deltas = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        return all(abs(d) < self.success_plateau_eps for d in deltas)

    def update(self, recent_rewards, success_rate=None):
        if success_rate is not None:
            self.success_hist.append(float(success_rate))

        x = np.asarray(list(recent_rewards), dtype=float)
            # self.pat_cnt = 0
            # # record diagnostics even on short samples
            # self.last_diag = {
            #     'CM/samples': int(len(x)),
            #     'CM/pat_cnt': int(self.pat_cnt),
            #     'CM/success_last': float(self.success_hist[-1]) if len(self.success_hist) else None,
            # }
            # return False

        w = x[-self.N:] if len(x) >= self.N else x
        ewma = self._ewma(w)
        trend_ok, slope, p95m5, eps_trend = self._trend_ok(ewma)
        var_ok, mean, std, cv_or_ratio = self._variance_ok(ewma)
        qual_ok = self._quality_ok()

        # ok = trend_ok and qual_ok
        ok = trend_ok and var_ok and qual_ok
        # if self.target_success is None:
        #     ok = ok and self._success_plateau()

        self.pat_cnt = self.pat_cnt + 1 if ok else 0

        # store diagnostics for wandb
        self.last_diag = {
            'CM/samples': int(len(w)),
            'CM/ewma_last': float(ewma[-1]),
            'CM/slope': float(slope),
            'CM/p95_minus_p5': float(p95m5),
            'CM/eps_trend': float(eps_trend),
            'CM/trend_check': bool(trend_ok),
            'CM/mean': float(mean),
            'CM/std': float(std),
            'CM/cv_or_std_over_iqr': float(cv_or_ratio),
            'CM/var_check': float(var_ok),
            'CM/quality_check': float(qual_ok),
            'CM/pat_cnt': int(self.pat_cnt),
            'CM/success_last': float(self.success_hist[-1]) if len(self.success_hist) else None,
        }

        return self.pat_cnt >= self.patience



# # --- videoskills/utils/convergence_monitor.py (instrumented for wandb diagnostics) ---
# import numpy as np
# from collections import deque
#
# class ConvergenceMonitor:
#     """
#     收敛判据：滑动窗口 W（N），将窗口一分为二，分别计算前半段/后半段的一阶线性斜率 s1/s2。
#     条件：|s1| < slope_thr 且 |s2| < slope_thr 且 |s1 - s2| < diff_thr
#     通过后计入 patience 计数；若设置了 target_success，则还需满足质量门槛；
#     若未设置 target_success，则使用 success 平台化检测作为兜底。
#     可选：对窗口序列做 EWMA(α) 平滑（alpha=None 表示不用平滑）。
#     """
#
#     def __init__(self,
#                  N=50,                 # 滑窗长度
#                  alpha=None,           # EWMA 平滑系数；None 表示不用平滑
#                  slope_thr=1e-3,       # 各半窗斜率绝对值阈值
#                  diff_thr=5e-4,        # 两半窗斜率差阈值
#                  patience=3,           # 连续满足次数
#                  target_success=None,  # 质量门槛，如 0.95；None 则用平台化兜底
#                  success_plateau_eps=0.03, # 平台判定阈值（相邻 success_rate 的绝对差）
#                  success_plateau_k=3        # 近 k+1 次评估都几乎不变 → 平台
#                  ):
#         self.N = int(N)
#         self.alpha = alpha
#         self.slope_thr = float(slope_thr)
#         self.diff_thr = float(diff_thr)
#         self.patience = int(patience)
#         self.target_success = None if target_success is None else float(target_success)
#         self.success_plateau_eps = float(success_plateau_eps)
#         self.success_plateau_k = int(success_plateau_k)
#
#         self.pat_cnt = 0
#         self.success_hist = deque(maxlen=10)  # 保存最近若干次 eval 的 success_rate
#         self.last_diag = {}                   # 暴露给 W&B 的诊断字段
#
#     # ---------- 工具函数 ----------
#
#     def _ewma(self, x):
#         """可选 EWMA 平滑；alpha=None 时直接返回原序列。"""
#         if self.alpha is None:
#             return np.asarray(x, dtype=float)
#         a = float(self.alpha)
#         y = []
#         m = float(x[0])
#         for r in x:
#             m = a * float(r) + (1 - a) * m
#             y.append(m)
#         return np.asarray(y, dtype=float)
#
#     def _linear_slope(self, y):
#         """简单一阶最小二乘斜率。"""
#         y = np.asarray(y, dtype=float)
#         n = len(y)
#         if n < 2:
#             return 0.0
#         t = np.arange(n, dtype=float)
#         t_mean = t.mean()
#         denom = ((t - t_mean) ** 2).sum() + 1e-12
#         slope = ((t - t_mean) * (y - y.mean())).sum() / denom
#         return float(slope)
#
#     def _pair_slopes(self, w):
#         """取窗口 w 的前半与后半，计算 s1/s2。"""
#         n = len(w)
#         h = n // 2
#         w1 = w[:h]
#         w2 = w[h:]
#         s1 = self._linear_slope(w1) if len(w1) >= 2 else 0.0
#         s2 = self._linear_slope(w2) if len(w2) >= 2 else 0.0
#         return float(s1), float(s2)
#
#     def _quality_ok(self):
#         """若设了 target_success，则以其为门槛；否则返回 True（交给平台化兜底）。"""
#         if self.target_success is None or len(self.success_hist) == 0:
#             return True
#         return float(self.success_hist[-1]) >= self.target_success
#
#     def _success_plateau(self):
#         """
#         未设 target_success 时的兜底平台检测：
#         最近 (k+1) 次 success_rate，两两差值的绝对值都 < eps → 认为平台化。
#         """
#         if len(self.success_hist) < self.success_plateau_k + 1:
#             return False
#         recent = list(self.success_hist)[-(self.success_plateau_k + 1):]
#         deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
#         return all(abs(d) < self.success_plateau_eps for d in deltas)
#
#     # ---------- 主接口 ----------
#
#     def update(self, recent_rewards, success_rate=None):
#         """
#         recent_rewards: 可迭代的最近奖励（建议传“新结束的 episodic returns”或全局 buffer 的后 N 个）
#         success_rate:   本次 eval 的成功率（0~1），用于质量门槛/平台检测
#         """
#         if success_rate is not None:
#             self.success_hist.append(float(success_rate))
#
#         x = np.asarray(list(recent_rewards), dtype=float)
#         min_needed = max(10, self.N // 2)  # 太短不判断
#         if len(x) < min_needed:
#             self.pat_cnt = 0
#             self.last_diag = {
#                 'CM/samples': int(len(x)),
#                 'CM/pat_cnt': int(self.pat_cnt),
#                 'CM/success_last': float(self.success_hist[-1]) if len(self.success_hist) else None,
#                 'CM/slope_ok_1': False,
#                 'CM/slope_ok_2': False,
#                 'CM/diff_ok': False,
#                 'CM/ok': 0.0,
#             }
#             return False
#
#         # 取滑窗
#         w_raw = x[-self.N:] if len(x) >= self.N else x
#         # 可选平滑
#         w = self._ewma(w_raw)
#
#         # 前后半窗斜率
#         s1, s2 = self._pair_slopes(w)
#         abs_s1 = abs(s1)
#         abs_s2 = abs(s2)
#         abs_diff = abs(s1 - s2)
#
#         slope_ok_1 = abs_s1 < self.slope_thr
#         slope_ok_2 = abs_s2 < self.slope_thr
#         diff_ok    = abs_diff < self.diff_thr
#
#         # 质量门槛
#         qual_ok = self._quality_ok()
#         ok = slope_ok_1 and slope_ok_2 and diff_ok and qual_ok
#
#         # 若未设置 target_success，则再要求 success 平台化（避免“低水平稳定”）
#         # if self.target_success is None:
#         #     ok = ok and self._success_plateau()
#
#         self.pat_cnt = self.pat_cnt + 1 if ok else 0
#
#         # 记录诊断字段（便于 wandb 面板调参）
#         self.last_diag = {
#             'CM/samples': int(len(w)),
#             'CM/window_last': float(w[-1]) if len(w) else None,
#             'CM/s1': float(s1),
#             'CM/s2': float(s2),
#             'CM/abs_s1': float(abs_s1),
#             'CM/abs_s2': float(abs_s2),
#             'CM/abs_diff': float(abs_diff),
#             'CM/slope_thr': float(self.slope_thr),
#             'CM/diff_thr': float(self.diff_thr),
#             'CM/slope_ok_1': bool(slope_ok_1),
#             'CM/slope_ok_2': bool(slope_ok_2),
#             'CM/diff_ok': bool(diff_ok),
#             'CM/quality_ok': bool(qual_ok),
#             'CM/pat_cnt': int(self.pat_cnt),
#             'CM/success_last': float(self.success_hist[-1]) if len(self.success_hist) else None,
#             'CM/ok': float(ok),
#         }
#
#         return self.pat_cnt >= self.patience

