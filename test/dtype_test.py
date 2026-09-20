"""Every advertised dtype on both GPU entry points, including returned dtypes."""

import numpy as np
import pytest

import jax.numpy as jnp

from helpers import skew
from pfcuda import pfaffian, slog_pfaffian

# dtype -> (dtype of log|Pf|, relative tolerance for that precision)
CASES = [
    pytest.param(np.float32, np.float32, 1e-3, id="float32"),
    pytest.param(np.float64, np.float64, 1e-8, id="float64"),
    pytest.param(np.complex64, np.float32, 1e-3, id="complex64"),
    pytest.param(np.complex128, np.float64, 1e-8, id="complex128"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype, real_dtype, rtol", CASES)
def test_pfaffian_dtype(dtype, real_dtype, rtol):
    A = skew(16, np.random.default_rng(5), dtype)
    pf = pfaffian(jnp.array(A))

    assert pf.dtype == np.dtype(dtype)
    assert np.isclose(
        np.asarray(pf, dtype=np.complex128) ** 2,
        np.linalg.det(A.astype(np.complex128)),
        rtol=rtol,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype, real_dtype, rtol", CASES)
def test_slog_pfaffian_dtype(dtype, real_dtype, rtol):
    A = skew(40, np.random.default_rng(5), dtype)
    log_abs, sign = slog_pfaffian(jnp.array(A))

    assert log_abs.dtype == np.dtype(real_dtype)
    assert sign.dtype == np.dtype(dtype)

    _, expected = np.linalg.slogdet(A.astype(np.complex128))
    assert np.isclose(2 * np.asarray(log_abs, dtype=np.float64), expected.real, rtol=rtol)


@pytest.mark.parametrize("dtype, real_dtype, rtol", CASES)
def test_slog_pfaffian_odd_dimension_dtype(dtype, real_dtype, rtol):
    """The odd-n shortcut builds its return values without calling the kernel."""
    A = skew(41, np.random.default_rng(5), dtype)
    log_abs, sign = slog_pfaffian(jnp.array(A))

    assert log_abs.dtype == np.dtype(real_dtype)
    assert sign.dtype == np.dtype(dtype)
    assert np.isneginf(log_abs)
    assert sign == 0
