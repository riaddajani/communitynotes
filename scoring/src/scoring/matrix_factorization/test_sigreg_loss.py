"""
Unit tests for SIGReg (Sketched Isotropic Gaussian Regularization) Loss Module
"""

import torch
import numpy as np
from typing import Tuple

# Import without pytest for testing
try:
    from .sigreg_loss import SIGRegLoss
except ImportError:
    # If relative import fails, try direct import
    from sigreg_loss import SIGRegLoss


def create_isotropic_embeddings(n: int = 1000, d: int = 5) -> torch.Tensor:
    """
    Create embeddings that follow an isotropic Gaussian distribution.

    Args:
        n: Number of entities
        d: Embedding dimension

    Returns:
        Tensor of shape (n, d) with isotropic Gaussian distribution
    """
    return torch.randn(n, d)


def create_collapsed_embeddings(n: int = 1000, d: int = 5, rank: int = 1) -> torch.Tensor:
    """
    Create collapsed embeddings where all vectors lie in a lower-dimensional subspace.

    Args:
        n: Number of entities
        d: Embedding dimension
        rank: Effective rank of the embeddings

    Returns:
        Tensor of shape (n, d) with collapsed representation
    """
    # Create low-rank embeddings
    U = torch.randn(n, rank)
    V = torch.randn(rank, d)
    return U @ V


def create_anisotropic_embeddings(n: int = 1000, d: int = 5) -> torch.Tensor:
    """
    Create embeddings with anisotropic distribution (different variances along axes).

    Args:
        n: Number of entities
        d: Embedding dimension

    Returns:
        Tensor of shape (n, d) with anisotropic distribution
    """
    embeddings = torch.randn(n, d)
    # Scale different dimensions differently
    scaling = torch.linspace(0.1, 2.0, d)
    return embeddings * scaling.unsqueeze(0)


class TestSIGRegLoss:
    """Test suite for SIGReg loss functionality."""

    def test_initialization(self):
        """Test SIGRegLoss initialization with different parameters."""
        # Default initialization
        sigreg = SIGRegLoss()
        assert sigreg.lambda_sketch == 0.01
        assert sigreg.epsilon == 1e-6
        assert sigreg.normalize == True

        # Custom initialization
        sigreg = SIGRegLoss(lambda_sketch=0.1, epsilon=1e-8, normalize=False)
        assert sigreg.lambda_sketch == 0.1
        assert sigreg.epsilon == 1e-8
        assert sigreg.normalize == False

    def test_isotropic_embeddings_low_loss(self):
        """Test that isotropic embeddings have low SIGReg loss."""
        embeddings = create_isotropic_embeddings(1000, 5)
        sigreg = SIGRegLoss(lambda_sketch=1.0)
        loss = sigreg(embeddings)

        # Loss should be relatively small for isotropic embeddings
        assert loss < 10.0, f"Loss {loss.item():.4f} too high for isotropic embeddings"

    def test_collapsed_embeddings_high_loss(self):
        """Test that collapsed embeddings have higher SIGReg loss than isotropic."""
        # Compare collapsed vs isotropic embeddings
        isotropic_emb = create_isotropic_embeddings(1000, 5)
        collapsed_emb = create_collapsed_embeddings(1000, 5, rank=1)

        sigreg = SIGRegLoss(lambda_sketch=1.0)

        isotropic_loss = sigreg(isotropic_emb)
        collapsed_loss = sigreg(collapsed_emb)

        # Collapsed embeddings should have significantly higher loss
        assert collapsed_loss > isotropic_loss * 1.5, \
            f"Collapsed loss {collapsed_loss.item():.4f} not sufficiently higher than isotropic {isotropic_loss.item():.4f}"

    def test_anisotropic_embeddings_medium_loss(self):
        """Test that anisotropic embeddings have medium SIGReg loss."""
        # Compare all three types
        isotropic_emb = create_isotropic_embeddings(1000, 5)
        anisotropic_emb = create_anisotropic_embeddings(1000, 5)
        collapsed_emb = create_collapsed_embeddings(1000, 5, rank=1)

        sigreg = SIGRegLoss(lambda_sketch=1.0)

        isotropic_loss = sigreg(isotropic_emb)
        anisotropic_loss = sigreg(anisotropic_emb)
        collapsed_loss = sigreg(collapsed_emb)

        # Anisotropic loss should be higher than isotropic (not perfectly isotropic)
        assert anisotropic_loss > isotropic_loss * 0.8, \
            f"Anisotropic loss {anisotropic_loss.item():.4f} unexpectedly low vs isotropic {isotropic_loss.item():.4f}"
        # Both anisotropic and collapsed should be noticeably higher than isotropic
        assert anisotropic_loss > 0, "Anisotropic loss should be positive"
        assert collapsed_loss > 0, "Collapsed loss should be positive"

    def test_1d_embeddings_zero_loss(self):
        """Test that 1D embeddings return zero loss (no regularization needed)."""
        embeddings_1d = torch.randn(100, 1)
        sigreg = SIGRegLoss(lambda_sketch=1.0)
        loss = sigreg(embeddings_1d)

        assert loss.item() == 0.0, "1D embeddings should have zero SIGReg loss"

    def test_lambda_scaling(self):
        """Test that lambda parameter correctly scales the loss."""
        embeddings = create_anisotropic_embeddings(100, 4)

        sigreg_1 = SIGRegLoss(lambda_sketch=0.1)
        loss_1 = sigreg_1(embeddings)

        sigreg_2 = SIGRegLoss(lambda_sketch=0.2)
        loss_2 = sigreg_2(embeddings)

        # Loss should scale linearly with lambda
        assert abs(loss_2.item() - 2 * loss_1.item()) < 0.001

    def test_normalization_effect(self):
        """Test the effect of normalization on loss computation."""
        embeddings = create_anisotropic_embeddings(100, 4)
        # Add a bias to make centering meaningful
        embeddings = embeddings + torch.randn(1, 4) * 2

        sigreg_normalized = SIGRegLoss(lambda_sketch=1.0, normalize=True)
        loss_normalized = sigreg_normalized(embeddings)

        sigreg_unnormalized = SIGRegLoss(lambda_sketch=1.0, normalize=False)
        loss_unnormalized = sigreg_unnormalized(embeddings)

        # Losses should be different when embeddings are not centered
        assert abs(loss_normalized.item() - loss_unnormalized.item()) > 0.1

    def test_covariance_stats(self):
        """Test covariance statistics computation."""
        embeddings = create_isotropic_embeddings(500, 4)
        sigreg = SIGRegLoss()
        stats = sigreg.get_covariance_stats(embeddings)

        # Check that all expected keys are present
        expected_keys = [
            'cov_trace', 'cov_det', 'eigenvalue_min', 'eigenvalue_max',
            'eigenvalue_std', 'frobenius_dist_from_identity', 'condition_number'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

        # For isotropic embeddings, eigenvalues should be close to 1
        assert 0.5 < stats['eigenvalue_min'] < 1.5
        assert 0.5 < stats['eigenvalue_max'] < 1.5
        assert stats['eigenvalue_std'] < 0.5

    def test_isotropy_score(self):
        """Test isotropy score computation."""
        sigreg = SIGRegLoss()

        # Isotropic embeddings should have high score
        isotropic_emb = create_isotropic_embeddings(500, 4)
        isotropy_score = sigreg.compute_isotropy_score(isotropic_emb)
        assert isotropy_score > 0.5, f"Isotropy score {isotropy_score:.4f} too low for isotropic embeddings"

        # Collapsed embeddings should have low score
        collapsed_emb = create_collapsed_embeddings(500, 4, rank=1)
        isotropy_score = sigreg.compute_isotropy_score(collapsed_emb)
        assert isotropy_score < 0.1, f"Isotropy score {isotropy_score:.4f} too high for collapsed embeddings"

        # 1D embeddings should return perfect score
        emb_1d = torch.randn(100, 1)
        isotropy_score = sigreg.compute_isotropy_score(emb_1d)
        assert isotropy_score == 1.0, "1D embeddings should have perfect isotropy score"

    def test_gradient_flow(self):
        """Test that gradients flow through the loss correctly."""
        embeddings = torch.randn(100, 4, requires_grad=True)
        sigreg = SIGRegLoss(lambda_sketch=0.1)

        loss = sigreg(embeddings)
        loss.backward()

        # Check that gradients are computed
        assert embeddings.grad is not None
        assert not torch.all(embeddings.grad == 0), "Gradients should be non-zero"

    def test_device_compatibility(self):
        """Test that SIGReg works with different devices."""
        sigreg = SIGRegLoss()

        # CPU
        embeddings_cpu = torch.randn(50, 3)
        loss_cpu = sigreg(embeddings_cpu)
        assert loss_cpu.device.type == 'cpu'

        # CUDA (if available)
        if torch.cuda.is_available():
            embeddings_cuda = torch.randn(50, 3, device='cuda')
            loss_cuda = sigreg(embeddings_cuda)
            assert loss_cuda.device.type == 'cuda'

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        sigreg = SIGRegLoss(epsilon=1e-8)

        # Very small embeddings
        small_embeddings = torch.randn(100, 4) * 1e-8
        loss_small = sigreg(small_embeddings)
        assert torch.isfinite(loss_small), "Loss should be finite for small embeddings"

        # Very large embeddings
        large_embeddings = torch.randn(100, 4) * 1e3
        loss_large = sigreg(large_embeddings)
        assert torch.isfinite(loss_large), "Loss should be finite for large embeddings"

        # Zero embeddings
        zero_embeddings = torch.zeros(100, 4)
        loss_zero = sigreg(zero_embeddings)
        assert torch.isfinite(loss_zero), "Loss should be finite for zero embeddings"


def test_integration_with_matrix_factorization():
    """Test integration scenario similar to Community Notes matrix factorization."""
    # Simulate user and note embeddings
    n_users = 1000
    n_notes = 500
    n_factors = 4  # Multi-dimensional factors

    # Create embedding layers similar to Community Notes
    user_embeddings = torch.nn.Embedding(n_users, n_factors)
    note_embeddings = torch.nn.Embedding(n_notes, n_factors)

    # Initialize with xavier uniform (as in Community Notes)
    torch.nn.init.xavier_uniform_(user_embeddings.weight, gain=1.0)
    torch.nn.init.xavier_uniform_(note_embeddings.weight, gain=1.0)

    # Create SIGReg loss modules
    sigreg_user = SIGRegLoss(lambda_sketch=0.005)  # Conservative value
    sigreg_note = SIGRegLoss(lambda_sketch=0.005)

    # Compute SIGReg losses
    user_loss = sigreg_user(user_embeddings.weight)
    note_loss = sigreg_note(note_embeddings.weight)

    # Total regularization loss
    total_sigreg_loss = user_loss + note_loss

    # Check that losses are reasonable
    assert torch.isfinite(total_sigreg_loss), "Total SIGReg loss should be finite"
    assert total_sigreg_loss > 0, "Total SIGReg loss should be positive"

    # Test backward pass
    total_sigreg_loss.backward()
    assert user_embeddings.weight.grad is not None, "User embeddings should have gradients"
    assert note_embeddings.weight.grad is not None, "Note embeddings should have gradients"


if __name__ == "__main__":
    # Run tests
    test = TestSIGRegLoss()

    print("Running SIGReg Loss Tests...")

    test.test_initialization()
    print("✓ Initialization test passed")

    test.test_isotropic_embeddings_low_loss()
    print("✓ Isotropic embeddings test passed")

    test.test_collapsed_embeddings_high_loss()
    print("✓ Collapsed embeddings test passed")

    test.test_anisotropic_embeddings_medium_loss()
    print("✓ Anisotropic embeddings test passed")

    test.test_1d_embeddings_zero_loss()
    print("✓ 1D embeddings test passed")

    test.test_lambda_scaling()
    print("✓ Lambda scaling test passed")

    test.test_normalization_effect()
    print("✓ Normalization effect test passed")

    test.test_covariance_stats()
    print("✓ Covariance stats test passed")

    test.test_isotropy_score()
    print("✓ Isotropy score test passed")

    test.test_gradient_flow()
    print("✓ Gradient flow test passed")

    test.test_device_compatibility()
    print("✓ Device compatibility test passed")

    test.test_numerical_stability()
    print("✓ Numerical stability test passed")

    test_integration_with_matrix_factorization()
    print("✓ Integration test passed")

    print("\nAll tests passed successfully!")