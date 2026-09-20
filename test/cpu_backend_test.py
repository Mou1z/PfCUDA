"""CPU backends: pfaffian_cpu (C++) and pfaffian_py (NumPy). No GPU needed."""

import numpy as np
import pytest

from helpers import known_pfaffian_case, sized, skew
from pfcuda import pfaffian_cpu, pfaffian_py

BACKENDS = [
    pytest.param(pfaffian_cpu, id="cpp"),
    pytest.param(pfaffian_py, id="numpy"),
]

SIZES = sized(fast={2, 4, 8, 32}, full=[2, 4, 8, 16, 32, 64, 128])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", SIZES)
def test_det_relation(backend, n):
    A = skew(n, np.random.default_rng(1))
    assert np.isclose(backend(A.copy()) ** 2, np.linalg.det(A), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", SIZES)
def test_matches_known_pfaffian(backend, n, samples):
    rng = np.random.default_rng(0)
    for _ in range(samples):
        A, expected = known_pfaffian_case(n, rng)
        assert np.isclose(backend(A.copy()), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_backends_agree(n, samples):
    rng = np.random.default_rng(7)
    for _ in range(samples):
        A = skew(n, rng)
        assert np.isclose(
            pfaffian_cpu(A.copy()), pfaffian_py(A.copy()), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_zero_matrix(backend, n):
    assert backend(np.zeros((n, n))) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_2x2_is_the_off_diagonal(backend):
    A = np.array([[0.0, 3.0], [-3.0, 0.0]])
    assert np.isclose(backend(A.copy()), 3.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_4x4_closed_form(backend):
    A = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, 0.0, 4.0, 5.0],
            [-2.0, -4.0, 0.0, 6.0],
            [-3.0, -5.0, -6.0, 0.0],
        ]
    )
    # 1*6 - 2*5 + 3*4 = 8
    assert np.isclose(backend(A.copy()), 8.0)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_odd_dimension_differs_between_backends(n):
    """pfaffian_py matches the GPU path and returns 0.0; pfaffian_cpu raises."""
    A = skew(n, np.random.default_rng(0))
    assert pfaffian_py(A.copy()) == 0.0
    with pytest.raises(RuntimeError, match="n must be even"):
        pfaffian_cpu(A.copy())


def test_cpu_is_float64_only_and_discards_imaginary_part():
    rng = np.random.default_rng(3)
    real = skew(8, rng)
    complexified = real.astype(np.complex128) + 1j * skew(8, rng)

    with pytest.warns(np.exceptions.ComplexWarning):
        got = pfaffian_cpu(complexified.copy())

    assert np.isclose(got, pfaffian_cpu(real.copy()), rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_is_overwritten_above_4x4(backend):
    A = skew(8, np.random.default_rng(0))
    working = A.copy()
    backend(working)
    assert not np.allclose(working, A)


@pytest.mark.parametrize("backend", BACKENDS)
def test_small_inputs_are_left_alone(backend):
    """n <= 4 takes a closed form rather than factoring."""
    A = skew(4, np.random.default_rng(0))
    working = A.copy()
    backend(working)
    assert np.allclose(working, A)
