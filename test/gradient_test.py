"""JVP rules for both GPU entry points, checked against finite differences.

Each distinct (dtype, size) pair costs ~1.5s of XLA compilation, so only one
float64 case per rule stays in the fast tier.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from helpers import skew
from pfcuda import pfaffian, slog_pfaffian

pytestmark = pytest.mark.gpu

slow = pytest.mark.slow

ALL_DTYPES = [
    pytest.param(np.float64, id="float64"),
    pytest.param(np.float32, id="float32", marks=slow),
    pytest.param(np.complex128, id="complex128", marks=slow),
    pytest.param(np.complex64, id="complex64", marks=slow),
]
REAL_DTYPES = [
    pytest.param(np.float64, id="float64"),
    pytest.param(np.float32, id="float32", marks=slow),
]
COMPLEX_DTYPES = [
    pytest.param(np.complex128, id="complex128"),
    pytest.param(np.complex64, id="complex64", marks=slow),
]

TOL = {np.float32: 2e-2, np.complex64: 2e-2, np.float64: 1e-6, np.complex128: 1e-6}


def _step(dtype):
    return 1e-4 if dtype in (np.float64, np.complex128) else 1e-2


def _directions(n, dtype, seed=3):
    rng = np.random.default_rng(seed)
    return jnp.array(skew(n, rng, dtype)), jnp.array(skew(n, rng, dtype) * 0.1)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("n", [8, pytest.param(16, marks=slow), pytest.param(32, marks=slow)])
def test_pfaffian_jvp_matches_finite_difference(dtype, n):
    A, V = _directions(n, dtype)
    _, tangent = jax.jvp(pfaffian, (A,), (V,))

    eps = _step(dtype)
    fd = (pfaffian(A + eps * V) - pfaffian(A - eps * V)) / (2 * eps)

    assert np.isclose(complex(tangent), complex(fd), rtol=TOL[dtype], atol=TOL[dtype])


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("n", [40, pytest.param(64, marks=slow)])
def test_slog_pfaffian_jvp_matches_finite_difference(dtype, n):
    A, V = _directions(n, dtype)
    (log_mag, sign), (d_log_mag, d_sign) = jax.jvp(slog_pfaffian, (A,), (V,))

    assert d_log_mag.dtype == log_mag.dtype
    assert d_sign.dtype == sign.dtype

    eps = _step(dtype)
    lo, _ = slog_pfaffian(A - eps * V)
    hi, _ = slog_pfaffian(A + eps * V)
    fd = (hi - lo) / (2 * eps)

    assert np.isclose(float(d_log_mag), float(fd), rtol=TOL[dtype], atol=TOL[dtype])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_real_input_leaves_the_sign_stationary(dtype):
    """For real matrices the sign is +-1, so its tangent must vanish."""
    A, V = _directions(40, dtype)
    _, (_, d_sign) = jax.jvp(slog_pfaffian, (A,), (V,))
    assert d_sign == 0


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_complex_input_rotates_the_phase(dtype):
    """For complex matrices the phase moves, so a zero tangent would be wrong."""
    A, V = _directions(40, dtype)
    _, (_, d_sign) = jax.jvp(slog_pfaffian, (A,), (V,))

    eps = _step(dtype)
    _, lo = slog_pfaffian(A - eps * V)
    _, hi = slog_pfaffian(A + eps * V)
    fd = (hi - lo) / (2 * eps)

    assert abs(complex(d_sign)) > 0
    assert np.isclose(complex(d_sign), complex(fd), rtol=TOL[dtype], atol=TOL[dtype])


@slow
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_grad_through_pfaffian(dtype):
    A, _ = _directions(16, dtype)
    g = jax.grad(pfaffian)(A)
    assert g.shape == (16, 16)
    assert np.all(np.isfinite(np.asarray(g)))
