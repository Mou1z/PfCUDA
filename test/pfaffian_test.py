"""GPU pfaffian(), matrices up to 32x32.

Only tests that reach a kernel are marked gpu; the size guard and the n <= 4
closed forms are pure Python.
"""

import numpy as np
import pytest

import jax.numpy as jnp
from pfapack.pfaffian import pfaffian as pfapack_pfaffian

from helpers import known_pfaffian_case, sized, skew
from pfcuda import pfaffian

SIZES = sized(fast={2, 4, 8, 16, 32}, full=range(2, 33, 2))


@pytest.mark.gpu
@pytest.mark.parametrize("n", sized(fast={2, 8, 18}, full=range(2, 20, 2)))
def test_zero_matrix(n):
    assert pfaffian(np.zeros((n, n))) == 0.0


@pytest.mark.parametrize("n", [1, 3, 5, 11, 15])
def test_odd_dimension(n):
    A = skew(n, np.random.default_rng(0))
    assert np.isclose(pfaffian(A), 0.0, atol=1e-12)


@pytest.mark.parametrize("a", [0.0, 1.0, -3.5, 1e-12, 1e10])
def test_2x2_is_the_off_diagonal(a):
    A = np.array([[0, a], [-a, 0]], dtype=np.float64)
    assert np.isclose(pfaffian(A), a)


def test_4x4_closed_form():
    A = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, 0.0, 4.0, 5.0],
            [-2.0, -4.0, 0.0, 6.0],
            [-3.0, -5.0, -6.0, 0.0],
        ]
    )
    # 1*6 - 2*5 + 3*4 = 8
    assert np.isclose(pfaffian(jnp.array(A)), 8.0)


@pytest.mark.gpu
@pytest.mark.parametrize("n", SIZES)
def test_det_relation(n):
    A = skew(n, np.random.default_rng(1))
    assert np.isclose(pfaffian(A) ** 2, np.linalg.det(A), rtol=1e-8, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("n", SIZES)
def test_matches_pfapack(n, samples):
    rng = np.random.default_rng(123)
    for i in range(samples):
        A = skew(n, rng)
        assert np.allclose(
            pfapack_pfaffian(A), pfaffian(jnp.array(A)), rtol=1e-8, atol=1e-10
        ), f"sample {i}"


@pytest.mark.gpu
@pytest.mark.parametrize("n", SIZES)
def test_matches_known_pfaffian(n, samples):
    rng = np.random.default_rng(0)
    for i in range(samples):
        A, expected = known_pfaffian_case(n, rng)
        assert np.isclose(
            pfaffian(A), expected, rtol=1e-10, atol=1e-12
        ), f"sample {i}"


def test_rejects_oversized_matrix():
    A = skew(34, np.random.default_rng(0))
    with pytest.raises(ValueError, match="maximum supported size"):
        pfaffian(jnp.array(A))


def test_rejects_unsupported_dtype():
    A = skew(8, np.random.default_rng(0)).astype(np.int64)
    with pytest.raises(TypeError, match="does not support"):
        pfaffian(jnp.array(A))
