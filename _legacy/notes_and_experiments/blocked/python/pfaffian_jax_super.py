import jax
from jax import Array
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

def pivot(data):
    A, Ak, V, W, i, j = data

    tmp = A[j]
    A = A.at[j].set(A[i])
    A = A.at[i].set(tmp)

    tmp = A[j]
    A = A.at[j].set(A[i])
    A = A.at[i].set(tmp)

    V = V.at[[i, j], :].set(V[[j, i], :])
    W = W.at[[i, j], :].set(W[[j, i], :])

    tmp = Ak[i]
    Ak = Ak.at[i].set(Ak[j])
    Ak = Ak.at[j].set(tmp)

    return A, Ak, V, W, -1

def no_pivot(data):
    A, Ak, V, W, _, _ = data
    return A, Ak, V, W, 1

@jax.jit
def pfaffian(A):
    n = A.shape[0]

    if n == 2:
        return A[0, 1]

    blockSize = (n - 2) // 2

    V = jnp.zeros((n, blockSize))
    W = jnp.zeros((n, blockSize))

    def dR(V, W, k):
        V0 = lax.dynamic_slice(V, [k, 0], (1, blockSize))
        W0 = lax.dynamic_slice(W, [k, 0], (1, blockSize))
        return jnp.ravel(V0 @ W.T - W0 @ V.T)

    def body(data, k):
        i = k // 2
        A, V, W = data

        idx = jnp.arange(n)
        mask = idx >= (k + 1)
        kRow = jnp.where(mask, A[k] + dR(V, W, k), 0.0)

        kp = jnp.abs(kRow).argmax()
        # A, kRow, V, W, sign = lax.cond(kp != k + 1, pivot, no_pivot, (A, kRow, V, W, k + 1, kp))
        pivot_element = kRow[k + 1] #* sign

        kCol = -(A[k + 1] + dR(V, W, k + 1))

        mask = idx >= (k + 2)
        V = V.at[:, i].set(jnp.where(mask, kRow / kRow[k + 1], 0.0))
        W = W.at[:, i].set(jnp.where(mask, kCol, 0.0))

        return (A, V, W), pivot_element

    (A, V, W), pivots = lax.scan(body, (A, V, W), jnp.arange(0, n - 1, 2))

    result = jnp.prod(pivots)

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

# print('Pfaffian:', pfaffian(A))