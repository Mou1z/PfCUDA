import jax
from jax import Array
import jax.numpy as jnp
from jax import lax
from jax import debug

import numpy as np
from pfapack.pfaffian import pfaffian as pf

# np.set_printoptions(
#     precision=3,
#     suppress=True,
#     linewidth=120,
#     threshold=1000
# )

def random_skew_symmetric_even(n, scale=1.0):
    if n % 2 != 0:
        raise ValueError("Matrix dimension n must be even.")

    M = scale * np.random.randn(n, n)
    A = M - M.T
    return A

jax.config.update("jax_enable_x64", True)

# def dEx(A, i, j, k):    
#     def true_fn(A, i, j, k):
#         return (
#             (A[k + 1, j] * A[k, i]) -
#             (A[k + 1, i] * A[k, j])
#         )

#     return lax.cond(
#         k < (i - (i % 2)),
#         lambda _: true_fn(A, i, j, k),
#         lambda _: 0.0,
#         operand=None
#     )

# def dE(A, n, i, j):
#     def true_fn(A, n, i, j):
#         z = jnp.arange(0, n, 2)
#         x = jnp.full_like(z, i)
#         y = jnp.full_like(z, j)
#         return jax.vmap(lambda xi, yi, zi: dEx(A, xi, yi, zi))(x, y, z).sum()

#     return lax.cond(
#         j > i,
#         lambda _: true_fn(A, n, i, j),
#         lambda _: 0.0,
#         operand=None
#     )

# def dR(A, n, k):
#     y = jnp.arange(n)
#     x = jnp.full_like(y, k)
#     return jax.vmap(lambda xi, yi: dE(A, n, xi, yi))(x, y)

# def pivot(data):
#     A, i, j = data
#     A = A.at[[i, j], :].set(A[[j, i], :])
#     A = A.at[:, [i, j]].set(A[:, [j, i]])
#     return A, -1

def pivot(A, i, j):
    temp = A[i]
    A = A.at[i].set(A[j])
    A = A.at[j].set(temp)

    temp = A[:, i]
    A = A.at[:, i].set(A[:, j])
    A = A.at[:, j].set(temp)

    return A, -1.0

def no_pivot(A, i, j):
    return A, 1.0

@jax.jit
def pfaffian(A):
    n = A.shape[0]

    def dR(V, W, k):
        i = k // 2
        r = n // 2
        
        mask = jnp.arange(r) < i

        V0 = jnp.where(mask, V[:, k], 0.0)
        W0 = jnp.where(mask, W[:, k], 0.0)

        V1 = jnp.where(mask[:, None], V, 0.0)
        W1 = jnp.where(mask[:, None], W, 0.0)

        return (V0 @ W1) - (W0 @ V1)

    def body(data, k):
        A, result = data
        
        V = A[0::2, :]
        W = A[1::2, :]

        A = A.at[k].set(A[k] + dR(V, W, k))

        mask = jnp.arange(n) > k
        Ak = A[k] * mask
        kp = jnp.abs(Ak).argmax()
        # A, sign = lax.cond(kp != k + 1, pivot, no_pivot, A, k + 1, kp)

        # result *= sign
        result *= A[k, k + 1]

        A = A.at[k].set(A[k] / A[k, k + 1])
        A = A.at[k + 1].set(-(A[k + 1] + dR(V, W, k + 1)))

        return (A, result), None

    data, _ = lax.scan(body, (A, 1), jnp.arange(0, n - 1, 2))
    A, result = data

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