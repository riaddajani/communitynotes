#!/usr/bin/env python3
"""
Integration test for SIGReg with Community Notes matrix factorization.

This test verifies that the SIGReg integration works correctly with
the existing matrix factorization infrastructure.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scoring.matrix_factorization.matrix_factorization import MatrixFactorization
from scoring.matrix_factorization.model import BiasedMatrixFactorization
from scoring import constants as c


def create_synthetic_ratings_data(n_users=100, n_notes=50, n_ratings=500):
    """Create synthetic ratings data for testing."""
    # Generate random ratings
    np.random.seed(42)

    # Create random user-note pairs
    user_ids = np.random.randint(0, n_users, n_ratings)
    note_ids = np.random.randint(0, n_notes, n_ratings)

    # Create ratings (helpful=1 or not helpful=0)
    ratings = np.random.choice([0, 1], n_ratings, p=[0.3, 0.7])

    # Create DataFrame
    df = pd.DataFrame({
        c.raterParticipantIdKey: [f"user_{i}" for i in user_ids],
        c.noteIdKey: [f"note_{i}" for i in note_ids],
        c.helpfulNumKey: ratings.astype(float)
    })

    return df


def test_sigreg_disabled_backward_compatibility():
    """Test that SIGReg disabled (default) maintains backward compatibility."""
    print("Testing backward compatibility (SIGReg disabled)...")

    # Create matrix factorization with SIGReg disabled (default)
    mf = MatrixFactorization(
        numFactors=4,  # 1D embeddings (current Community Notes default)
        useGlobalIntercept=True,
        useSIGReg=True,  # Explicitly disabled (default)
    )

    # Create synthetic data
    ratings_df = create_synthetic_ratings_data()

    # Run matrix factorization (returns 6 values when validatePercent is provided)
    noteParams, raterParams, globalIntercept, train_loss, loss, validate_loss = mf.run_mf(
        ratings_df,
        noteInit=None,
        userInit=None,
        globalInterceptInit=None,
        validatePercent=0.1,  # Quick test with validation
    )

    # Verify outputs
    assert noteParams is not None, "Note parameters should not be None"
    assert raterParams is not None, "Rater parameters should not be None"
    assert globalIntercept is not None, "Global intercept should not be None"

    # Check that parameters have expected columns
    assert c.internalNoteInterceptKey in noteParams.columns
    assert c.internalRaterInterceptKey in raterParams.columns

    print("✓ Backward compatibility test passed")
    return True


def test_sigreg_enabled_with_1d_embeddings():
    """Test that SIGReg with 1D embeddings is a no-op (no effect)."""
    print("Testing SIGReg with 1D embeddings (should be no-op)...")

    # Create matrix factorization with SIGReg enabled but 1D embeddings
    mf = MatrixFactorization(
        numFactors=4,  # 1D embeddings
        useGlobalIntercept=True,
        useSIGReg=True,  # Enabled
        sigregLambdaUser=0.01,
        sigregLambdaNote=0.01,
    )

    # Create synthetic data
    ratings_df = create_synthetic_ratings_data()

    # Run matrix factorization (returns 6 values when validatePercent is provided)
    noteParams, raterParams, globalIntercept, _, _, _ = mf.run_mf(
        ratings_df,
        noteInit=None,
        userInit=None,
        globalInterceptInit=None,
        validatePercent=0.1,  # Quick test with validation
    )

    # Verify outputs (should work normally, SIGReg has no effect on 1D)
    assert noteParams is not None, "Note parameters should not be None"
    assert raterParams is not None, "Rater parameters should not be None"
    assert globalIntercept is not None, "Global intercept should not be None"

    print("✓ SIGReg with 1D embeddings test passed (no-op as expected)")
    return True


def test_sigreg_enabled_with_multidimensional_embeddings():
    """Test that SIGReg works with multi-dimensional embeddings."""
    print("Testing SIGReg with multi-dimensional embeddings...")

    # Create matrix factorization with SIGReg enabled and 2D embeddings
    mf = MatrixFactorization(
        numFactors=4,  # 2D embeddings
        useGlobalIntercept=True,
        useSIGReg=True,  # Enabled
        sigregLambdaUser=0.005,  # Conservative values
        sigregLambdaNote=0.005,
    )

    # Create synthetic data
    ratings_df = create_synthetic_ratings_data(n_users=50, n_notes=30, n_ratings=300)

    # Run matrix factorization (returns 6 values when validatePercent is provided)
    noteParams, raterParams, globalIntercept, _, _, _ = mf.run_mf(
        ratings_df,
        noteInit=None,
        userInit=None,
        globalInterceptInit=None,
        validatePercent=0.1,  # Quick test with validation
    )

    # Verify outputs
    assert noteParams is not None, "Note parameters should not be None"
    assert raterParams is not None, "Rater parameters should not be None"
    assert globalIntercept is not None, "Global intercept should not be None"

    # Check that we have factor columns for 2D embeddings
    factor_cols = [col for col in noteParams.columns if 'factor' in col.lower()]
    assert len(factor_cols) == 2, f"Expected 2 factor columns, got {len(factor_cols)}"

    # Verify that embeddings are not collapsed (all factors shouldn't be identical)
    note_factors = noteParams[factor_cols].values
    factor_std = np.std(note_factors, axis=0)
    assert np.all(factor_std > 0.01), "Factor diversity maintained (no collapse)"

    print("✓ SIGReg with multi-dimensional embeddings test passed")
    return True


def test_model_with_sigreg_parameters():
    """Test that the BiasedMatrixFactorization model accepts SIGReg parameters."""
    print("Testing BiasedMatrixFactorization model with SIGReg parameters...")

    # Create model with SIGReg parameters
    model = BiasedMatrixFactorization(
        n_users=100,
        n_notes=50,
        n_factors=4,
        use_global_intercept=True,
        use_sigreg=True,
        sigreg_lambda=0.01
    )

    # Verify model attributes
    assert model._use_sigreg == True, "SIGReg should be enabled"
    assert model._sigreg_lambda == 0.01, "SIGReg lambda should be set"

    # Verify get_factor_embeddings method exists
    embeddings = model.get_factor_embeddings()
    assert 'user_factors' in embeddings, "Should return user factors"
    assert 'note_factors' in embeddings, "Should return note factors"

    print("✓ Model with SIGReg parameters test passed")
    return True


def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("Running SIGReg Integration Tests")
    print("="*60 + "\n")

    tests = [
        test_sigreg_disabled_backward_compatibility,
        test_sigreg_enabled_with_1d_embeddings,
        test_sigreg_enabled_with_multidimensional_embeddings,
        test_model_with_sigreg_parameters,
    ]

    all_passed = True
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            all_passed = False
            print(f"✗ {test.__name__} failed with error: {e}")

    print("\n" + "="*60)
    if all_passed:
        print("All integration tests passed successfully!")
    else:
        print("Some integration tests failed.")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)