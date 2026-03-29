import jax
from jax import Array
import jax.numpy as jnp
from jax import lax

import numpy as np
from pfapack.pfaffian import pfaffian as pf

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

jax.config.update("jax_enable_x64", True)

def pivot(data):
    A, V, W, i, j = data
    
    A = A.at[[i, j], :].set(A[[j, i], :])
    A = A.at[:, [i, j]].set(A[:, [j, i]])

    V = V.at[[i, j], :].set(V[[j, i], :])
    W = W.at[[i, j], :].set(W[[j, i], :])

    return A, V, W, -1

def no_pivot(data):
    A, V, W, _, _ = data
    return A, V, W, 1

@jax.jit
def pfaffian(A):
    n = A.shape[0]
    blockSize = (n - 2) // 2

    V = jnp.zeros((n, blockSize))
    W = jnp.zeros((n, blockSize))

    def dR(V, W, k):
        m = V.shape[1]
        i = k // 2

        mask = (jnp.arange(m) < i).astype(V.dtype)

        Vk = V[k] * mask
        Wk = W[k] * mask

        return Vk @ W.T - Wk @ V.T

    def body(data, k):
        i = k // 2
        A, V, W, result = data

        R1 = (A[k] + dR(V, W, k))

        # kp = jnp.abs(R1).argmax()
        kp = k + 1
        result *= R1[kp]
        V = V.at[:, i].set((R1 / R1[kp]))

        # A, V, W, sign = lax.cond(
        #     (kp != k + 1), 
        #     pivot, no_pivot, 
        #     (A, V, W, k + 1, kp)
        # )
        # result *= sign
        
        R2 = (A[:, k + 1] - dR(V, W, k + 1))
        W = W.at[:, i].set(R2)

        return (A, V, W, result), None

    (_, V, W, result), _ = lax.scan(body, (A, V, W, 1.0), jnp.arange(0, n, 2))

    return result
    
A = jnp.array([
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
], dtype = jnp.float64)

# A = random_skew_symmetric_even(10, 100)

print()
print('Pfaffian:', pfaffian(A.copy()))
print('Expected:', pf(A.copy()))