"""TCEC, as written in the paper. Independent of the quantization engine.

Single-step error model      eps_t = K_t * mu_q               (Eq. 14)
Closed-form scaling          K_{t,i} from Eq. 17, gamma = 1e-8
Regularizer                  lambda_1 = 0.01 * mean(mu_q^2) / var(mu)
Correction (m = 1)           x_{t-1} -= B_t K_t mu_q(t)
                                      + sqrt(abar_t / abar_{t-1}) B_{t+1} K_{t+1} mu_q(t+1)   (Eq. 18)
"""
import torch

GAMMA = 1e-8
MODES = ("none", "local", "paper")  # local = current step only (Eq. 14), paper = Eq. 18


class KStats:
    """Accumulates Eq. 17 per (step, channel). mu_q and mu must come from the same x_t."""

    def __init__(self, steps, channels):
        shape = (steps, channels)
        self.qq = torch.zeros(shape, dtype=torch.float64)
        self.qf = torch.zeros(shape, dtype=torch.float64)
        self.f = torch.zeros(shape, dtype=torch.float64)
        self.ff = torch.zeros(shape, dtype=torch.float64)
        self.n = torch.zeros(shape, dtype=torch.float64)

    @torch.no_grad()
    def update(self, i, mu_q, mu):
        dims = (0,) + tuple(range(2, mu.dim()))
        q, f = mu_q.detach().double().cpu(), mu.detach().double().cpu()
        self.qq[i] += q.pow(2).sum(dims)
        self.qf[i] += (q * f).sum(dims)
        self.f[i] += f.sum(dims)
        self.ff[i] += f.pow(2).sum(dims)
        self.n[i] += q.numel() / q.shape[1]

    def solve(self):
        var_f = (self.ff / self.n - (self.f / self.n).pow(2)).clamp(min=1e-12)
        lam = 0.01 * (self.qq / self.n) / var_f
        return ((self.qq - self.qf) / (self.qq + lam + GAMMA)).float()


class Corrector:
    """Holds mu_q(t+1) between steps. Call reset() before each sampling run."""

    def __init__(self, K, mode):
        assert mode in MODES
        self.K, self.mode = K, mode
        self.reset()

    def reset(self):
        self.i, self.prev = 0, None

    @torch.no_grad()
    def apply(self, x_prev, mu_q, B, w):
        """x_prev = uncorrected x_{t-1}; B = B_t; w = sqrt(abar_t / abar_{t-1})"""
        i, self.i = self.i, self.i + 1
        if self.mode == "none":
            return x_prev
        K = self.K[i].to(mu_q).view(1, -1, *[1] * (mu_q.dim() - 2))
        cur = B * K * mu_q
        out = x_prev - cur
        if self.mode == "paper" and self.prev is not None:
            out = out - w * self.prev
        self.prev = cur
        return out
