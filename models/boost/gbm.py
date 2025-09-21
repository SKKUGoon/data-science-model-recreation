from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted  # type: ignore


@dataclass
class _TreeSpec:
    """Specification of a single weak learner (decision tree) in a GBM model. + Feature subset mask."""
    tree: DecisionTreeRegressor
    feature_idx: np.ndarray  # columns used by this tree (feature subsampling)


class _BaseGBM(BaseEstimator):
    """
    A minimum GBM scaffold
    - bootstrap row sampling
    - column (feature) subsampling) (colsample)
    - storing weak learners and their feature subsets
    """

    def __init__(
        self,
        n_estimators: int = 300,  # number of boosting rounds. (iterative round)
        learning_rate: float = 0.05,  # shrinkage applied to each tree's output
        max_depth: int = 3,  # depth of individual regression trees
        min_samples_leaf: int = 1,  # minimum samples in a leaf
        subsample: float = 1.0,  # row subsampling rate
        colsample: float = 1.0,  # feature subsampling rate
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,  # stop if validation loss doesn't improve for this many rounds
        validation_fraction: float = 0.1,  # fraction of training data set aside for vlidation if early stopping is used
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ):
        assert 0 < subsample <= 1.0
        assert 0 < colsample <= 1.0
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.dtype = dtype
        self.device = device

        self._rng = np.random.RandomState(random_state)
        self._learners: List[_TreeSpec] = []
        self._fitted: bool = False
        self._n_features_: Optional[int] = None
        self._best_iteration_: Optional[int] = None
    
    def _row_sample_idx(self, n_sample: int) -> np.ndarray:
        if self.subsample > 1.0:
            return np.arange(n_sample)
        else:
            m = max(1, int(np.floor(self.subsample * n_sample)))
            return self._rng.choice(n_sample, size=m, replace=False)
        
    def _col_sample_idx(self, n_feature: int) -> np.ndarray:
        if self.colsample >= 1.0:
            return np.arange(n_feature)
        else:
            m = max(1, int(np.floor(self.colsample * n_feature)))
            return self._rng.choice(n_feature, size=m, replace=False)
        
    def _make_tree(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self._rng.randint(0, 2**31 -1)
        )
    
    def _init_model_output(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_init_model_output must be implemented in subclass")
    
    def _compute_psuedo_residual(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_compute_pseudo_residual must be implemented in subclass")
    
    def _loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_loss must be implemented in subclass")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_fitted")
        assert self._n_features_ is not None
        if X.shape[1] != self._n_features_:
            raise ValueError(f"Number of features in input ({X.shape[1]}) does not match training data ({self._n_features_})")
        
        y_pred = np.zeros(X.shape[0], dtype=np.float32) + self._init_model_output(torch.zeros(1, dtype=self.dtype, device=self.device)).item()
        for spec in self._learners:
            tree_pred = spec.tree.predict(X[:, spec.feature_idx])
            y_pred += self.learning_rate * tree_pred
        return y_pred
    
    @property
    def best_iteration_(self) -> Optional[int]:
        """The iteration with the best validation score (if early stopping is used)."""
        return self._best_iteration_ if self.early_stopping_rounds is not None else len(self._learners)
    
    def _split_train_valid(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            random: bool = True  # False if no data shuffling is desired (e.g. time series data)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.early_stopping_rounds is None:
            return X, y, X, y  # dummy return values
        n = X.shape[0]
        n_val = max(1, int(np.floor(self.validation_fraction * n)))
        if random:
            idx = torch.randperm(n, device=X.device)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
        else:
            idx = torch.arange(n, device=X.device)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class GBMRegressor(_BaseGBM):
    def _init_model_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        Initialization step for GBM in a regression case. 
        """
        f0 = y.mean()  # (1) compute the average of all targets
        self._f0 = float(f0.item())  # (2) store it as a Python float baseline
        return torch.full_like(y, fill_value=self._f0)  # Return a tensor filled with f0

    def _compute_pseudo_residual(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the pseudo-residuals for regression.
        """
        return y - y_pred  # Negative gradient of squared error loss
    
    def _loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss (mean squared error) for regression. MSE / 2
        """
        return torch.mean((y - y_pred) ** 2) * 0.5  # MSE / 2
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "GBMRegressor":
        X_t = X.to(device=self.device, dtype=self.dtype)
        y_t = y.to(device=self.device, dtype=self.dtype)
        _n, d = X_t.shape
        self._n_features_ = d
        self._learners = []
        self._fitted = False
        
        # Split
        X_tr, y_tr, X_val, y_val = self._split_train_valid(X_t, y_t, random=True)

        f_tr = self._init_model_output(y_tr)  # Initial prediction - all means
        if self.early_stopping_rounds is not None:
            # Create a separate f for validation to tract loss while not training on it
            f_val = torch.full(
                (X_val.shape[0], ), 
                fill_value=self._f0, 
                device=self.device, 
                dtype=self.dtype
            )
            best_val = float(self._loss(y_val, f_val).item())
            best_iter = 0
            rounds_since_best = 0

        for m in range(self.n_estimators):
            # Compute residuals
            r_tr = self._compute_pseudo_residual(y_tr, f_tr)

            # Sample rows and cols
            row_idx = self._row_sample_idx(X_tr.shape[0])
            col_idx = self._col_sample_idx(d)

            # Fit regression tree
            tree = self._make_tree()
            X_sub_np = X_tr[row_idx][:, col_idx].cpu().numpy()
            r_sub_np = r_tr[row_idx].cpu().numpy()
            tree.fit(X_sub_np, r_sub_np)

            # Update train predictions
            incr_tr = torch.tensor(tree.predict(X_tr[:, col_idx].cpu().numpy()), device=self.device, dtype=self.dtype)
            f_tr += f_tr + self.learning_rate * incr_tr

            # Track learner
            self._learners.append(_TreeSpec(tree=tree, feature_idx=col_idx))

            # early stopping
            if self.early_stopping_rounds is not None:
                incr_val = torch.tensor(tree.predict((X_val[:, col_idx].cpu().numpy())), dtype=self.dtype, device=self.device)
                f_val = f_val + self.learning_rate * incr_val  # type: ignore
                cur_val = float(self._loss(y_val, f_val).item())
                if cur_val + 1e-12 < best_val:  # type: ignore
                    best_val = cur_val
                    best_iter = m + 1
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1  # type: ignore
                    if rounds_since_best >= self.early_stopping_rounds:
                        # trim surplus learners
                        self._learners = self._learners[:best_iter]  # type: ignore
                        self._best_iteration_ = best_iter  # type: ignore
                        self._fitted = True
                        return self
        self._best_iteration_ = len(self._learners)
        self._fitted = True
        return self
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        X_np = X.cpu().numpy()
        return self._predict_raw(X_np)
