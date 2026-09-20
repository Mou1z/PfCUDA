"""Matrix construction shared by the test modules."""

import numpy as np
import pytest


def sized(fast, full):
    """Parametrize over `full`, marking every size outside `fast` as slow."""
    return [
        n if n in fast else pytest.param(n, marks=pytest.mark.slow) for n in full
    ]


def skew(n, rng, dtype=np.float64):
    """Random A == -A.T. Complex dtypes are not conjugated."""
    A = rng.normal(size=(n, n))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        A = A + 1j * rng.normal(size=(n, n))
    A = A.astype(dtype)
    return A - A.T


def _pfaffian_tridiag(upper_diag):
    n = len(upper_diag) + 1
    if n % 2 == 1:
        return 0.0
    return np.prod(upper_diag[0::2])


def _orthogonal(n, rng):
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    return Q * np.sign(np.diag(R))


def known_pfaffian_case(n, rng):
    """A dense skew matrix and its exact Pfaffian.

    B @ T @ B.T for orthogonal B and tridiagonal T, so Pf is det(B) * Pf(T) and
    Pf(T) is a product of entries -- a reference that cannot itself be wrong.
    """
    alt = (1.0 + rng.normal(size=(n - 1,))) * (1 - np.arange(n - 1) % 2)
    T = np.diag(alt, k=1) - np.diag(alt, k=-1)
    B = _orthogonal(n, rng)
    return B @ T @ B.T, np.linalg.det(B) * _pfaffian_tridiag(alt)
