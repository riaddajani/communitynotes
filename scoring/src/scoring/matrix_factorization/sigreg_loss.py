"""
SIGReg (Sketched Isotropic Gaussian Regularization) Loss Module

This module implements the SIGReg loss from the LeJEPA paper to enforce
isotropic Gaussian distribution of embeddings, preventing representation
collapse in matrix factorization models.

Reference: LeJEPA - Latent-Euclidean Joint-Embedding Predictive Architecture (Meta AI)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) for embedding layers.

    Enforces that embeddings follow an isotropic Gaussian distribution by
    regularizing the empirical covariance matrix towards identity.

    The loss is computed as:
    L_SIGReg = λ_sketch * ||Sketch(Z) - I||_F^2

    Where:
    - Z = embedding matrix (users or notes)
    - Sketch(Z) = Z^T @ Z / n (empirical covariance)
    - I = identity matrix (isotropic target)
    - ||·||_F = Frobenius norm
    """

    def __init__(
        self,
        lambda_sketch: float = 0.01,
        epsilon: float = 1e-6,
        normalize: bool = True
    ):
        """
        Initialize SIGReg loss module.

        Args:
            lambda_sketch: Weight for SIGReg loss term
            epsilon: Small constant for numerical stability
            normalize: Whether to normalize embeddings before computing covariance
        """
        super().__init__()
        self.lambda_sketch = lambda_sketch
        self.epsilon = epsilon
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for embedding matrix.

        Args:
            embeddings: Tensor of shape (n_entities, embedding_dim)

        Returns:
            SIGReg loss scalar
        """
        # Handle 1D embeddings (no regularization needed)
        if embeddings.ndim == 1 or embeddings.shape[1] == 1:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        # embeddings: (n, d)
        n, d = embeddings.shape

        # Optional normalization (center embeddings)
        if self.normalize:
            embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute empirical covariance: Sketch(Z) = Z^T @ Z / n
        # This is more efficient than computing full covariance matrix
        cov = (embeddings.T @ embeddings) / (n + self.epsilon)  # (d, d)

        # Target: identity matrix (isotropic Gaussian)
        identity = torch.eye(d, device=embeddings.device, dtype=embeddings.dtype)

        # Frobenius norm squared: ||Sketch(Z) - I||_F^2
        loss = torch.sum((cov - identity) ** 2)

        return self.lambda_sketch * loss

    def get_covariance_stats(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute diagnostic statistics for monitoring embedding quality.

        Args:
            embeddings: Tensor of shape (n_entities, embedding_dim)

        Returns:
            Dictionary with covariance matrix statistics
        """
        # Handle 1D embeddings
        if embeddings.ndim == 1 or embeddings.shape[1] == 1:
            return {
                'cov_trace': 1.0,
                'cov_det': 1.0,
                'eigenvalue_min': 1.0,
                'eigenvalue_max': 1.0,
                'eigenvalue_std': 0.0,
                'frobenius_dist_from_identity': 0.0,
                'condition_number': 1.0
            }

        n, d = embeddings.shape

        if self.normalize:
            embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

        cov = (embeddings.T @ embeddings) / (n + self.epsilon)

        # Compute eigenvalues to check isotropy
        # Move to CPU for eigenvalue computation if on MPS device (not yet supported)
        original_device = cov.device
        if cov.is_mps:
            cov_cpu = cov.cpu()
            eigenvalues = torch.linalg.eigvalsh(cov_cpu).to(original_device)
        else:
            eigenvalues = torch.linalg.eigvalsh(cov)

        # Condition number (ratio of largest to smallest eigenvalue)
        condition_number = eigenvalues.max() / (eigenvalues.min() + self.epsilon)

        # Compute determinant (may also need CPU fallback for MPS)
        if cov.is_mps:
            cov_det = torch.det(cov.cpu()).item()
        else:
            cov_det = torch.det(cov).item()

        return {
            'cov_trace': torch.trace(cov).item(),
            'cov_det': cov_det,
            'eigenvalue_min': eigenvalues.min().item(),
            'eigenvalue_max': eigenvalues.max().item(),
            'eigenvalue_std': eigenvalues.std().item(),
            'frobenius_dist_from_identity': torch.norm(
                cov - torch.eye(d, device=embeddings.device),
                p='fro'
            ).item(),
            'condition_number': condition_number.item()
        }

    def compute_isotropy_score(self, embeddings: torch.Tensor) -> float:
        """
        Compute isotropy score between 0 and 1 (1 = perfectly isotropic).

        Based on the ratio of minimum to maximum eigenvalues of the
        covariance matrix. A value close to 1 indicates isotropic distribution.

        Args:
            embeddings: Tensor of shape (n_entities, embedding_dim)

        Returns:
            Isotropy score between 0 and 1
        """
        if embeddings.ndim == 1 or embeddings.shape[1] == 1:
            return 1.0

        n, d = embeddings.shape

        if self.normalize:
            embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

        cov = (embeddings.T @ embeddings) / (n + self.epsilon)

        # Move to CPU for eigenvalue computation if on MPS device (not yet supported)
        if cov.is_mps:
            eigenvalues = torch.linalg.eigvalsh(cov.cpu()).to(cov.device)
        else:
            eigenvalues = torch.linalg.eigvalsh(cov)

        # Isotropy score: min_eigenvalue / max_eigenvalue
        isotropy = (eigenvalues.min() / (eigenvalues.max() + self.epsilon)).item()

        return max(0.0, min(1.0, isotropy))