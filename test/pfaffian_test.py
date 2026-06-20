import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pfapack.pfaffian import pfaffian as pfapack_pfaffian
from pfcuda import pfaffian

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
    for n in range(2, 20, 2):
        A = np.zeros((n, n))
        pf = pfaffian(A)

        assert pf == 0.0

def test_pfaffian_odd_dimension():
    rng = np.random.default_rng(0)

    for n in [1, 3, 5, 11, 15]:
        A = rng.normal(size=(n, n))
        A = A - A.T  # skew-symmetric

        pf = pfaffian(A)

        assert np.isclose(pf, 0.0, atol=1e-12)

def test_pfaffian_2x2():
    values = [0.0, 1.0, -3.5, 1e-12, 1e10]

    for a in values:
        A = np.array([[0, a], [-a, 0]], dtype=np.float64)
        pf = pfaffian(A)

        assert np.isclose(pf, a)

def test_pfaffian_det_relation():
    rng = np.random.default_rng(1)

    for n in range(2, 33, 2):
        A = rng.normal(size=(n, n))
        A = A - A.T

        pf = pfaffian(A)
        det = np.linalg.det(A)

        assert np.isclose(pf * pf, det, rtol=1e-8, atol=1e-10)

def test_pfaffian_random():
    rng = np.random.default_rng(123)

    for n in range(2, 33, 2):
        for _ in range(20):
            A = rng.normal(size=(n, n))
            A = A - A.T

            pf_ref = pfapack_pfaffian(A)
            pf_test = pfaffian(jnp.array(A))

            assert np.allclose(pf_ref, pf_test, rtol=1e-8, atol=1e-10)

def test_pfaffian_by_identity():
    rng = np.random.default_rng(seed=0)

    for n in range(2, 33, 2):
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

            calculated_pfaffian = pfaffian(B @ A @ B.T)
            expected_pfaffian = np.linalg.det(B) * pf_A

            assert np.isclose(
                expected_pfaffian,
                calculated_pfaffian,
                rtol=1e-10,
                atol=1e-12
            )