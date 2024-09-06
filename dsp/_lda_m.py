# Modified LDA

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from scipy.optimize import minimize

@dataclass
class LinearDiscriminantAnalysis_Regularized:
    scaling_: Optional[float] = field(default=None, init=False)
    W: Optional[np.ndarray] = field(default=None, init=False)  # Transformation matrix
    n_components: Optional[int] = field(default=None, init=True)

    def fit(self, X: np.ndarray, y: np.ndarray, lam = 1.0):
        """
        Fit the LDA model with regularization to the data X and labels y.
        """
        class_labels = np.unique(y)
        
        if self.n_components is None:
            self.n_components = min(X.shape[1], len(class_labels) - 1)
        
        mean_vectors = []
        for cls in class_labels:
            mean_vectors.append(np.mean(X[y == cls], axis=0))
        
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cls, mean_vec in zip(class_labels, mean_vectors):
            class_scatter = np.cov(X[y == cls].T)
            S_W += class_scatter
        
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == class_labels[i]].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            overall_mean = overall_mean.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        W = []
        for _ in range(self.n_components):
            w_init = np.random.rand(X.shape[1])
            result = minimize(custom_lda_loss, w_init, args=(S_W, S_B, lam), method='BFGS')
            w_opt = result.x.reshape(-1, 1)
            W.append(w_opt)
            S_B = S_B - S_B @ w_opt @ w_opt.T @ S_B / (w_opt.T @ S_W @ w_opt + lam)
        
        self.W = np.hstack(W)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data X using the fitted LDA model.
        """
        if self.W is None:
            raise ValueError("The model has not been fitted yet.")
        return X.dot(self.W)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
        """
        Fit the LDA model and then transform the input data X.
        """
        self.fit(X, y, lam)
        return self.transform(X)

# Custom LDA loss function as discussed before
def custom_lda_loss(w, S_W, S_B, lambda_reg):
    w = w.reshape(-1, 1)
    numerator = w.T @ S_B @ w
    denominator = w.T @ (S_W + lambda_reg * np.eye(S_W.shape[0])) @ w
    return -float(numerator / denominator)  # We minimize, so negate
