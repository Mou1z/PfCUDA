import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

@jax.jit
def pfaffian(A):
    result = 1.0
    n = A.shape[0]

    if n == 2:
        return A[0, 1]

    V = jnp.zeros((n, n - 2))
    W = jnp.zeros((n, n - 2))

    def calc_vector_blocks(i, data):
        k = i * 2
        A, V, W = data

        uAk = A[k] + ((V[k] @ W.T) - (W[k] @ V.T))

        idx = jnp.arange(n)
        mask = idx >= (k + 2)
        V = V.at[:, k].set(jnp.where(mask, uAk / uAk[k+1], 0.0))
        W = W.at[:, k].set(A[:, k + 1] + ((W[k + 1] @ V.T) - (V[k + 1] @ W.T)))

        return A, V, W

    _, V, W = lax.fori_loop(0, (n // 2) - 1, calc_vector_blocks, (A, V, W))

    VWt = V @ W.T
    fA = A + (VWt - VWt.T)
    
    result = jnp.prod(fA[jnp.arange(0, n, 2), jnp.arange(1, n, 2)])

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
])

print('Pfaffian:', pfaffian(A))