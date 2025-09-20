from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PCA:
    n_components: int  # Target dimensionality
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    center: bool = True  # PCA assumes centered data

    # fitted
    components_: Optional[torch.Tensor] = None  # Principal axes. 
    singular_values_: Optional[torch.Tensor] = None  # SVD singular values (scale of variance)
    explained_variance_: Optional[torch.Tensor] = None  # Variance per components
    explained_variance_ratio_: Optional[torch.Tensor] = None  
    mean_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "PCA":
        # Centering + SVD to get principal axes
        X_ = X.to(dtype=self.dtype, device=self.device)
        n, d = X_.shape
        if self.center:
            self.mean_ = X_.mean(0, keepdim=True)
            Xc = X_ - self.mean_
        else:
            self.mean_ = torch.zeros((1, d), dtype=self.dtype, device=self.device)
            Xc = X_
        
        # Economy SVD: Xc = U S Vh; columns of V are principal directions
        
        # SVD: For any matrix n*m, SVD factors it into U (n*n), S (n*m), Vh (m*m)
        # U: left singular vectors (orthonormal)
        # S: singular values (non-negative, sorted)
        # Vh: right singular vectors (orthonormal)
        # It turns any data matrix into rotation (V) + scaling (S) + rotation (U)

        # X.T @ X (feature-feature covariance) -> Eigen decomposition
        # X @ X.T (sample-sample covariance)

        # Image X unit sphere
        # V -> Which direction the sphere is rotated
        # S -> How the sphere is stretched into ellipsoid
        # U -> How stretched ellipsoid is oriented

        # PCA uses V (right singular vectors) and S (singular values)
        # V will be the principal axes (directions of max variance)
        # S tells us variance along each axis. 

        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # type: ignore
        k = min(self.n_components, Vh.shape[0])  # type: ignore 
        
        self.components_ = Vh[:k]  # type: ignore      (k, d) top-k principal axes
        self.singular_values_ = S[:k]  # type: ignore  (k,) scale along components

        self.explained_variance_ = (S[:k] ** 2) / max(n - 1, 1)
        total_var = (Xc.pow(2).sum(0) / max(n - 1, 1)).sum().clamp(min=1e-15)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var  # type: ignore
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.components_ is not None, "PCA not fitted"
        X_ = X.to(dtype=self.dtype, device=self.device)
        return (X_ - self.mean_) @ self.components_.T  # type: ignore
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        assert self.components_ is not None, "PCA not fitted"
        Z_ = Z.to(dtype=self.dtype, device=self.device)
        return Z_ @ self.components_ + self.mean_  # type: ignore
    

        
        
