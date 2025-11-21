from .bayesian_mlp import BayesianMLP
from .bayesian_linear import normal_log_prob
import torch

# Implement negative ELBO loss function
# ELBO: Evidence Lower Bound

def elbo_regression(
    model: BayesianMLP,
    xb: torch.Tensor,
    yb: torch.Tensor,
    observed_sigma: float,
    n_batches: int,
    mc_samples: int,
) -> torch.Tensor:
    """
    Loss = NLL + KL / N_BATCHES
    """
    device = xb.device
    nll = torch.zeros((), device=device)
    kl = torch.zeros((), device=device)
    observed_sigma_t = torch.tensor(observed_sigma, device=device)

    for i in range(n_batches):
        pred, kl_s = model(xb, sample=True)
        nll += (-normal_log_prob(yb, pred, observed_sigma_t)).mean()
        kl += kl_s

    nll /= mc_samples
    kl /= mc_samples
    return nll + kl / n_batches