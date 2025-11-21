import math
from typing import Optional, Tuple

from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from torch import nn


class BNNConfig(BaseModel):
    prior_sigma: float = Field(1.0, description="Standard deviation of the prior distribution")
    observed_sigma: float = Field(0.1, description="Assumed observation noise std. Gaussian NLML loss")
    
    hidden_dim: int = Field(64, description="Hidden dimension of the neural network")
    epochs: int = Field(100, description="Number of epochs to train the model")
    batch_size: int = Field(128, description="Batch size for training")
    lr: float = Field(0.001, description="Learning rate for training")

    # In BBN, each weight is not fixed but follows a distribution:
    # w ~ N(mu, sigma)
    # During forward pass, we sample weights from this distribution 
    # Because of that random sampling, the model's output also becomes stochastic
    # Changes slightly each time we sample a new set of weights
    mc_train_samples: int = Field(10, description="Number of Monte Carlo samples for training")
    mc_test_samples: int = Field(100, description="Number of Monte Carlo samples for testing")

    device: str = Field("mps", description="Device to use for training (Macbook Apple Silicon)")


def normal_log_prob(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    p(x) = Probability density function of the normal distribution

    returns log(p(x)) Log of the probability density function

    Why use log?
    - Multiplying probabilities can lead to underflow(very small numbers)
    - Log-space turn products to sum. 
    """
    var = std ** 2 + 1e-12  # Add small epsilon to avoid log(0)
    return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mu) ** 2 / var)


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Use nn.Parameter to register as model parameters 
        # that will be optimized during training

        # y = Wx + b
        # W ~ N(mu, sigma)
        # b ~ N(mu, sigma)

        # Variational params for weights
        # rho is the parameter of the variance
        # variance = softplus(rho) + 1e-12
        # softplus(rho) = log(1 + exp(rho)).
        #  - If the rho is negative(small), the variance is close to 0
        #  - If the rho is positive(large), the variance is close to infinity
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))  # Normal distribution of mean 0 and std 0.1
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-5, -4))  # Uniform distribution between -5 and -4

        # Variational params for bias
        # Separate the parameters for bias.
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).uniform_(-5, -4))

        # Fixed zero-mean: Assume Gaussian: prior
        self.register_buffer("prior_mu", torch.zeros(1))
        self.register_buffer("prior_sigma", torch.tensor(prior_sigma))

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _sample_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w_sigma = self._softplus(self.weight_rho)
        b_sigma = self._softplus(self.bias_rho)

        # Reparametreization trick
        # torch.randn_like(w_sigma): Creates a random noise from standard normal distribution
        # Add that to the mean + std * err
        W = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)

        return W, b

    def _kl(self, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w_sigma = self._softplus(self.weight_rho)
        b_sigma = self._softplus(self.bias_rho)
        
        # True posterior: p(w|D)
        # Simpler approximation: q(w)
        # Minimize the KL divergence between q(w) and p(w|D)
        # We know the X. Calculate the log probability from p and q.
        # Compare the log probability
        log_qw = normal_log_prob(W, self.weight_mu, w_sigma).sum()
        log_qb = normal_log_prob(b, self.bias_mu, b_sigma).sum()

        log_pw = normal_log_prob(W, self.prior_mu, self.prior_sigma).sum()
        log_pb = normal_log_prob(b, self.prior_mu, self.prior_sigma).sum()

        return (log_qw - log_pw) + (log_qb - log_pb)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Params:
            sample: bool, whether that layer uses stochastic sampling or deterministic mode
                - True: Stochastic. Each forward pass gets different random weights. Full Bayesian inference
                - False: Deterministic. Always uses the same weights. Uses "best guess" 
        """
        if sample:
            W, b = self._sample_params()
        else:
            W, b = self.weight_mu, self.bias_mu
        
        out = F.linear(x, W, b)
        kl = self._kl(W, b)
        return out, kl


if __name__ == "__main__":
    cfg = BNNConfig()
    torch.manual_seed(42)

    layer = BayesianLinear(
        in_features=3,
        out_features=2,
        prior_sigma=cfg.prior_sigma
    )

    x = torch.randn(4, 3)

    out1, kl1 = layer(x, sample=True)
    out2, kl2 = layer(x, sample=True)

    out_det, kl_det = layer(x, sample=False)

    print("=== Input ===")
    print(x)
    print("\n=== Stochastic forward pass #1 ===")
    print(out1)
    print(f"KL divergence term: {kl1.item():.4f}")

    print("\n=== Stochastic forward pass #2 ===")
    print(out2)
    print(f"KL divergence term: {kl2.item():.4f}")

    print("\n=== Deterministic forward (using posterior means) ===")
    print(out_det)
    print(f"KL divergence term (still computed): {kl_det.item():.4f}")

    # Check difference between stochastic passes
    diff = (out1 - out2).abs().mean()
    print(f"\nAverage |difference| between stochastic samples: {diff.item():.6f}")