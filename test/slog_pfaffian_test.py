import pytest
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pfapack.pfaffian import pfaffian as pfapack_pfaffian
from pfcuda import slog_pfaffian

def pfaffian_tridiag(upper_diag):
    n = len(upper_diag) + 1
    if n % 2 == 1:
        return 0.0
    return np.prod(upper_diag[0::2])

def create_orthogonal_matrix_np(n, rng):
    A = rng.normal(size=(n, n))

    Q, R = np.linalg.qr(A)

    d = np.diag(R)
    Q = Q * np.sign(d)

    return Q

def test_pfaffian_zero_matrix():
    for n in range(34, 68, 2):
        A = np.zeros((n, n))
        logabs, sign = slog_pfaffian(jnp.array(A))

        assert sign == 0
        assert np.isneginf(logabs)

def test_pfaffian_odd_dimension():
    rng = np.random.default_rng(0)

    for n in [35, 37, 39]:
        A = rng.normal(size=(n, n))
        A = A - A.T

        logabs, sign = slog_pfaffian(jnp.array(A))

        assert sign == 0
        assert np.isneginf(logabs)

def test_pfaffian_det_relation():
    rng = np.random.default_rng(1)

    for n in range(34, 66, 2):
        A = rng.normal(size=(n, n))
        A = A - A.T

        det = np.linalg.det(A)

        logabs, sign = slog_pfaffian(jnp.array(A))
        pf_sq = np.exp(2 * logabs)

        assert np.isclose(pf_sq, det, rtol=1e-8, atol=1e-10)

def test_pfaffian_random():
    rng = np.random.default_rng(123)

    for n in range(40, 300, 10):
        for _ in range(20):
            A = rng.normal(size=(n, n))
            A = A - A.T

            pf_ref = pfapack_pfaffian(A)

            logabs, sign = slog_pfaffian(jnp.array(A))
            pf_test = sign * np.exp(logabs)

            assert np.allclose(pf_ref, pf_test, rtol=1e-8, atol=1e-10)

def test_pfaffian_by_identity():
    rng = np.random.default_rng(seed=0)

    for n in range(40, 300, 10):
        for _ in range(10):
            main_diagonal = np.zeros(n)

            alt_diagonal = 1.0 + rng.normal(size=(n - 1,))

            mask = 1 - (np.arange(n - 1) % 2)
            alt_diagonal = alt_diagonal * mask

            A = (
                np.diag(alt_diagonal, k=1)
                + np.diag(main_diagonal)
                - np.diag(alt_diagonal, k=-1)
            )

            pf_A = pfaffian_tridiag(alt_diagonal)

            B = create_orthogonal_matrix_np(n, rng)

            expected_pfaffian = np.linalg.det(B) * pf_A
            logabs, sign = slog_pfaffian(B @ A @ B.T)
            calculated_pfaffian = sign * np.exp(logabs)

            assert np.isclose(
                expected_pfaffian,
                calculated_pfaffian,
                rtol=1e-10,
                atol=1e-12
            )