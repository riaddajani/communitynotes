"""
Sanity Checks for Community Notes Scoring Models

This package contains standalone sanity check scripts for validating
matrix factorization model outputs.

Available checks:
- sanity_check_6factor.py: Checks for 6-factor matrix factorization model
"""

from .sanity_check_6factor import (
    run_all_checks,
    check_predictive_performance,
    check_factor_variance,
    check_factor_orthogonality,
)

__all__ = [
    "run_all_checks",
    "check_predictive_performance",
    "check_factor_variance",
    "check_factor_orthogonality",
]
