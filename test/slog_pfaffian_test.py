"""GPU slog_pfaffian(), matrices of 34x34 and larger.

Only tests that reach a kernel are marked gpu; the size guard and the odd-n
shortcut are pure Python.
"""

import numpy as np
import pytest

import jax.numpy as jnp
from pfapack.pfaffian import pfaffian as pfapack_pfaffian

from helpers import known_pfaffian_case, sized, skew
from pfcuda import slog_pfaffian

# Fewer sizes than the pfaffian() sweep: these dominate the suite's runtime.
LARGE_SIZES = sized(fast={40, 100, 200}, full=range(40, 300, 10))


@pytest.mark.gpu
@pytest.mark.parametrize("n", sized(fast={34, 48, 66}, full=range(34, 68, 2)))
def test_zero_matrix(n):
    log_abs, sign = slog_pfaffian(jnp.array(np.zeros((n, n))))
    assert sign == 0
    assert np.isneginf(log_abs)


@pytest.mark.parametrize("n", [35, 37, 39])
def test_odd_dimension(n):
    A = skew(n, np.random.default_rng(0))
    log_abs, sign = slog_pfaffian(jnp.array(A))
    assert sign == 0
    assert np.isneginf(log_abs)


@pytest.mark.gpu
@pytest.mark.parametrize("n", sized(fast={34, 48, 64}, full=range(34, 66, 2)))
def test_det_relation(n):
    A = skew(n, np.random.default_rng(1))
    log_abs, _ = slog_pfaffian(jnp.array(A))
    assert np.isclose(np.exp(2 * log_abs), np.linalg.det(A), rtol=1e-8, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("n", LARGE_SIZES)
def test_matches_pfapack(n, samples):
    rng = np.random.default_rng(123)
    for i in range(samples):
        A = skew(n, rng)
        log_abs, sign = slog_pfaffian(jnp.array(A))
        assert np.allclose(
            pfapack_pfaffian(A), sign * np.exp(log_abs), rtol=1e-8, atol=1e-10
        ), f"sample {i}"


@pytest.mark.gpu
@pytest.mark.parametrize("n", LARGE_SIZES)
def test_matches_known_pfaffian(n, samples):
    rng = np.random.default_rng(0)
    for i in range(samples):
        A, expected = known_pfaffian_case(n, rng)
        log_abs, sign = slog_pfaffian(A)
        assert np.isclose(
            sign * np.exp(log_abs), expected, rtol=1e-10, atol=1e-12
        ), f"sample {i}"


def test_rejects_undersized_matrix():
    A = skew(32, np.random.default_rng(0))
    with pytest.raises(ValueError, match="minimum supported size"):
        slog_pfaffian(jnp.array(A))


def test_rejects_unsupported_dtype():
    A = skew(40, np.random.default_rng(0)).astype(np.int64)
    with pytest.raises(TypeError, match="does not support"):
        slog_pfaffian(jnp.array(A))
