import numpy as np
from collections import deque

class ConvergenceMonitor:
    def __init__(self, N=100, alpha=0.15, cv_thr=0.08, trend_scale=1e-3,
                 patience=3, target_success=None, success_plateau_eps=0.01, success_plateau_k=3):
        self.N = N
        self.alpha = alpha
        self.cv_thr = cv_thr
        self.trend_scale = trend_scale
        self.patience = patience
        self.target_success = target_success
        self.success_plateau_eps = success_plateau_eps
        self.success_plateau_k = success_plateau_k
        self.pat_cnt = 0
        self.success_hist = deque(maxlen=10)

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
        return abs(slope) < eps_trend

    def _variance_ok(self, ewma):
        mean = float(np.mean(ewma)); std = float(np.std(ewma))
        if abs(mean) > 1e-3:
            cv = std / (abs(mean) + 1e-6)
            return cv < self.cv_thr
        else:
            iqr = np.percentile(ewma, 75) - np.percentile(ewma, 25)
            return std / (iqr + 1e-6) < 0.5

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
        if len(x) < max(10, self.N//2):
            self.pat_cnt = 0
            return False

        w = x[-self.N:] if len(x) >= self.N else x
        ewma = self._ewma(w)
        ok = self._trend_ok(ewma) and self._variance_ok(ewma) and self._quality_ok()

        if self.target_success is None:
            ok = ok and self._success_plateau()

        self.pat_cnt = self.pat_cnt + 1 if ok else 0
        return self.pat_cnt >= self.patience
