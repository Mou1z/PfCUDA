import numpy as np
from pfapack.pfaffian import pfaffian as pf
# from pfaffian_module import pfaffian as pf

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=120,
    threshold=1000
)

def random_skew_symmetric_even(n, scale=1.0):
    if n % 2 != 0:
        raise ValueError("Matrix dimension n must be even.")

    M = scale * np.random.randn(n, n)
    A = M - M.T
    return A

def pfaffian(A):
    n = A.shape[0]
    result = 1.0

    block_size = 32
    V = np.zeros((block_size, n))
    W = np.zeros((block_size, n))

    for k in range(0, n - 1):
        i = (k // 2) % block_size
        Ak = A[k, k + 1:]

        if k % 2 == 0:
            kp = k + 1 + np.abs(Ak).argmax()

            if A[k, kp] == 0:
                return 0.0

            if kp != k + 1:
                A[[k + 1, kp], :] = A[[kp, k + 1], :]
                A[:, [k + 1, kp]] = A[:, [kp, k + 1]]

                V[:i, [k + 1, kp]] = V[:i, [kp, k + 1]]
                W[:i, [k + 1, kp]] = W[:i, [kp, k + 1]]

                result *= -1

            result *= Ak[0]
            V[i, k + 2:] = Ak[1:] / Ak[0]
        else:
            W[i, k + 1:] = -Ak

        dVW = (V.T @ W - W.T @ V)

        if k != 0 and k % block_size == 0:
            dVW = (V.T @ W - W.T @ V)[k+1:,k+1:]
            A[k+1:,k+1:] += dVW
            A[k+1:,k+1:] -= dVW.T

        else:
            k += 1
            i = k // 2
            A[k, k + 1:] += (V.T[k, :i] @ W[:i, k + 1:] - W.T[k, :i] @ V[:i, k + 1:]))
    return result

A = np.array([
    [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9],
    [ -1,   0,  11,  12,  13,  14,  15,  16,  17,  18],
    [ -2, -11,   0,  23,  24,  25,  26,  27,  28,  29],
    [ -3, -12, -23,   0,  34,  35,  36,  37,  38,  39],
    [ -4, -13, -24, -34,   0,  45,  46,  47,  48,  49],
    [ -5, -14, -25, -35, -45,   0,  56,  57,  58,  59],
    [ -6, -15, -26, -36, -46, -56,   0,  67,  68,  69],
    [ -7, -16, -27, -37, -47, -57, -67,   0,  78,  79],
    [ -8, -17, -28, -38, -48, -58, -68, -78,   0,  89],
    [ -9, -18, -29, -39, -49, -59, -69, -79, -89,   0]
], dtype=np.float64)

# A = random_skew_symmetric_even(50, 0.1)

print()
print('Pfaffian:', pfaffian(A.copy()))
print('Expected:', pf(A.copy()))