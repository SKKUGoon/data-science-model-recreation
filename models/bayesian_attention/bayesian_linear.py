import math
import torch
from torch import nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class BayesSAConfig(BaseModel):
    dim: int = Field(..., description="Embedding dimension (input = output)")
    dropout: float = Field(0.1, description="Dropout rate")
    prior_sigma: float = Field(0.5, description="Prior sigma for the weights")
    init_rho: float = Field(-4.0, description="Initial rho for the variance. Later map to variance using softplus.")
    eps: float = Field(1e-12, description="Small epsilon to avoid log(0)")
    device: str = Field("mps", description="Device to use for training (Macbook Apple Silicon)")


class BayesianLinear(nn.Module):
    """
    Variational linear layer with Gaussian posterior q(W|theta) = N(mu, diag(sigma^2))
    Supports local reparameterization trick for lower-variance activations
    """
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        init_rho: float = -3.0,
        use_local_reparam: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_local_reparam = use_local_reparam  # Difference. Linear is used inside a larger stochastic structure. So we must manage variance explosion. Stabilzing the gradients
        self.eps = eps

        # variatonal parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.2))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), init_rho))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), init_rho))  # full must be tuples of ints. 

        # fixed zero-mean Gaussian prior
        self.register_buffer("prior_mu", torch.zeros(out_features, in_features))
        ...
    ...