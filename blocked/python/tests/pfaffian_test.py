import jax
import pytest
import numpy as np
import jax.numpy as jnp
from scipy.linalg import schur

from pfaffian_jax import pfaffian as pfaffian_py

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

def create_orthogonal_matrix_jpy(n, key):
    A = jax.random.normal(key, (n, n))

    Q, R = jnp.linalg.qr(A)
    
    d = jnp.diag(R)
    Q = Q * jnp.sign(d)
    
    return Q

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