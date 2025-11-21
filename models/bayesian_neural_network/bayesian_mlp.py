import torch
from torch import nn
import torch.nn.functional as F
from .bayesian_linear import BayesianLinear


class BayesianMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, prior_sigma: float):
        super().__init__()
        self.b1 = BayesianLinear(in_dim, hidden_dim, prior_sigma)
        self.b2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma)
        self.b3 = BayesianLinear(hidden_dim, out_dim, prior_sigma)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        kl_total = torch.zeros((), device=x.device)  # single zero tensor

        h, kl1 = self.b1(x, sample)  # run
        kl_total += kl1
        h = F.gelu(h)

        h, kl2 = self.b2(h, sample)
        kl_total += kl2
        h = F.gelu(h)

        y, kl3 = self.b3(h, sample)
        kl_total += kl3

        return y, kl_total

