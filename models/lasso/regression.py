import torch
from torch import nn
import numpy as np


# Ridge regression has a closed form solution
# But here implement it with gradient descent. 
class LASSORegression(nn.Module):
    def __init__(self, n_features: int, alpha: float = 1.0) -> None:
        """
        Ridge Regression model. Linear model with L2 penalty.

        Args:
            n_features (int): Number of input features. 
            alpha: regularization strength (default=1.0).
        """
        super().__init__() # type: ignore

        self.linear = nn.Linear(n_features, 1, bias=True)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def lasso_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        l1_penalty = self.alpha * torch.sum(torch.abs(self.linear.weight))  

        # Use L1 penalty terms on the weights
        # Effectively making them zero
        return mse_loss + l1_penalty

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
    
    @staticmethod
    def mae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y_hat - y)) 
    
    @staticmethod
    def r2_score(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        0 <= R^2 <= 1, 1 is perfect prediction.
             Happens when ss_residual is zero.
        1 <= Model is no better than just predicting the mean. residual is equal to total
        """
        ss_total = torch.sum((y - torch.mean(y)) ** 2)  # How far the data deviates from the mean
        ss_residual = torch.sum((y - y_hat) ** 2)  # Errors left inside the model
        return 1 - (ss_residual / ss_total)  # type: ignore
    
    @torch.inference_mode()
    def predict(self, x: torch.Tensor, to_numpy: bool = False) -> torch.Tensor | np.ndarray:
        was_training = self.training
        try:
            self.eval()
            y_hat = self.forward(x)
            return y_hat.detach().cpu().numpy() if to_numpy else y_hat
        finally:
            if was_training:
                self.train()
    