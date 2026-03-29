import pytest
import numpy as np

from pfaffian_op_s import pfaffian as pfaffian_py
# from pfapack.pfaffian import pfaffian as pfaffian_py

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

@pytest.mark.monkey
def test_pfaffian_py_by_identity():
    rng = np.random.default_rng(seed=0)

    for n in range(2, 50, 2):
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

            print(A)

            pf_A = pfaffian_tridiag(alt_diagonal)

            B = create_orthogonal_matrix_np(n, rng)

            calculated_pfaffian = pfaffian_py(B @ A @ B.T)
            expected_pfaffian = np.linalg.det(B) * pf_A

            assert np.isclose(
                expected_pfaffian,
                calculated_pfaffian,
                rtol=1e-10,
                atol=1e-12
            )
