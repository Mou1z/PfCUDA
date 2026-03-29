import jax
import pytest
import numpy as np
import jax.numpy as jnp
from scipy.linalg import schur
# from piquasso._math.pfaffian import pfaffian as pfaffian_cpp
# from piquasso._math.jax.pfaffian import pfaffian as pfaffian_py

from basic.pfaffian import pfaffian as pfaffian_py

def pfaffian_naive(matrix):
    if matrix.shape[0] == 0:
        return 1.0

    if matrix.shape[0] % 2 == 1:
        return 0.0

    blocks, O = schur(matrix)
    a = np.diag(blocks, 1)[::2]

    return np.prod(a) * np.linalg.det(O)

def pfaffian_tridiag(upper_diag):
    n = len(upper_diag) + 1
    if n % 2 == 1:
        return 0.0
    return np.prod(upper_diag[0::2]) 

def create_orthogonal_matrix_np(n, seed=42):
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    
    d = np.diag(R)
    Q *= np.sign(d)
    
    return Q

def create_orthogonal_matrix_jpy(n, key):
    A = jax.random.normal(key, (n, n))

    Q, R = jnp.linalg.qr(A)
    
    d = jnp.diag(R)
    Q = Q * jnp.sign(d)
    
    return Q

def random_skew_symmetric_matrix(n, key):
    A = jax.random.normal(key, (n, n))
    A = A - A.T
    return A

def pf_4x4(A):
    a, b, c = A[0,1], A[0,2], A[0,3]
    d, e, f = A[1,2], A[1,3], A[2,3]
    return a*f - b*e + c*d

def tangent_4x4(A, dA):
    a, b, c = A[0,1], A[0,2], A[0,3]
    d, e, f = A[1,2], A[1,3], A[2,3]
    return (
        dA[0,1]*f + a*dA[2,3]
        - dA[0,2]*e - b*dA[1,3]
        + dA[0,3]*d + c*dA[1,2]
    )

def test_pfaffian_empty():
    matrix = np.empty((0, 0), dtype=float)
    assert np.isclose(pfaffian_cpp(matrix), 1.0)

def test_pfaffian_2_by_2_skew_symmetric_float32():
    matrix = np.array(
        [
            [0, 1.7],
            [-1.7, 0],
        ],
        dtype=np.float32,
    )

    result = pfaffian_naive(matrix)
    output = pfaffian_cpp(matrix)

    assert output.dtype == np.float32

    assert np.isclose(output, result)
    assert np.isclose(output, 1.7)


def test_pfaffian_2_by_2_skew_symmetric_float64():
    matrix = np.array(
        [
            [0, 1.7],
            [-1.7, 0],
        ],
        dtype=np.float64,
    )

    output = pfaffian_cpp(matrix)

    assert output.dtype == np.float64

    assert np.isclose(output.item(), pfaffian_naive(matrix))
    assert np.isclose(output.item(), 1.7)


@pytest.mark.monkey
def test_pfaffian_4_by_4_skew_symmetric_random():
    for _ in range(100):

        A = np.random.rand(4, 4)
        matrix = A - A.T

        result = pfaffian_naive(matrix)
        output = pfaffian_cpp(matrix)

        assert np.isclose(output, result)


@pytest.mark.monkey
def test_pfaffian_6_by_6_skew_symmetric_random():
    for _ in range(100):

        A = np.random.rand(6, 6)
        matrix = A - A.T

        result = pfaffian_naive(matrix)
        output = pfaffian_cpp(matrix)

        assert np.isclose(output, result)

@pytest.mark.monkey
def test_pfaffian_cpp_by_identity():
    for n in range(2, 50, 2):
        for _ in range(10):
            main_diagonal = np.zeros(n)
            alt_diagonal = 1 + np.random.normal(size=(n - 1,))

            mask = 1 - (np.arange(n - 1) % 2)
            alt_diagonal = alt_diagonal * mask

            A = (
                np.diag(alt_diagonal, k=1) +
                np.diag(main_diagonal) -
                np.diag(alt_diagonal, k=-1)
            )

            pf_A = pfaffian_tridiag(alt_diagonal)

            B = create_orthogonal_matrix_np(n)

            calculated_pfaffian = pfaffian_cpp(B @ A @ B.T)
            expected_pfaffian = np.linalg.det(B) * pf_A

            assert np.isclose(expected_pfaffian, calculated_pfaffian, rtol=1e-3, atol=1e-6)

@pytest.mark.monkey
def test_pfaffian_py_by_identity():
    key = jax.random.PRNGKey(0)
    
    for n in range(2, 50, 2):
        for _ in range(10):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            main_diagonal = jnp.zeros(n)
            alt_diagonal = 1 + jax.random.normal(subkey1, (n - 1,))

            mask = 1 - (jnp.arange(n - 1) % 2)
            alt_diagonal = alt_diagonal * mask

            A = (
                jnp.diag(alt_diagonal, k = 1) +
                jnp.diag(main_diagonal) -
                jnp.diag(alt_diagonal, k = -1)
            )

            pf_A = pfaffian_tridiag(alt_diagonal)

            B = create_orthogonal_matrix_jpy(n, subkey2)

            calculated_pfaffian = pfaffian_py(B @ A @ B.T)
            expected_pfaffian = jnp.linalg.det(B) * pf_A

            assert jnp.isclose(expected_pfaffian, calculated_pfaffian)

@pytest.mark.monkey
def test_pfaffian_jvp_small():
    key = jax.random.PRNGKey(0)

    for _ in range(10):
        f32_max = jnp.finfo(jnp.float32).max
        a_unit = jax.random.uniform(key, (), minval=1e-7, maxval=1.0)
        a = a_unit * f32_max

        b_unit = jax.random.uniform(key, (), minval=1e-7, maxval=1.0)
        b = b_unit * f32_max

        A = jnp.array([
            [0.0,  a], 
            [-a, 0.0]
        ])

        dA = jnp.array ([
            [0.0,  b],
            [-b, 0.0]
        ])

        matrix_inverse = jnp.array([[0.0, -1.0/a], [1.0/a, 0.0]])
        trace = jnp.trace(matrix_inverse @ dA)

        expected_pfaffian = a
        expected_tangent = 0.5 * a * trace

        y, tan = jax.jvp(pfaffian_py, (A,), (dA,))

        assert np.allclose(y, expected_pfaffian)
        assert np.allclose(tan, expected_tangent)

@pytest.mark.monkey
def test_pfaffian_jvp_medium():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    
    A = random_skew_symmetric_matrix(4, key1)
    dA = random_skew_symmetric_matrix(4, key2)
    
    pf_exact = pf_4x4(A)
    tangent_exact = tangent_4x4(A, dA)
    
    y, tan = jax.jvp(pf_4x4, (A,), (dA,))
    
    assert np.allclose(y, pf_exact)
    assert np.allclose(tan, tangent_exact)