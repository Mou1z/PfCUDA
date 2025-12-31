import jax
from jax import Array
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

@jax.jit
def pfaffian(A):
    n = A.shape[0]

    if n == 2:
        return A[0, 1]

    blockSize = (n - 2) // 2

    V = jnp.zeros((n, blockSize))
    W = jnp.zeros((n, blockSize))

    def calc_vector_blocks(data, k):
        i = k // 2
        V, W = data

        uAk = A[k] + ((V[k] @ W.T) - (W[k] @ V.T))

        idx = jnp.arange(n)
        mask = idx >= (k + 2)
        V = V.at[:, i].set(jnp.where(mask, uAk / uAk[k+1], 0.0))
        W = W.at[:, i].set(A[:, k + 1] + ((W[k + 1] @ V.T) - (V[k + 1] @ W.T)))

        return (V, W), None

    (V, W), _ = lax.scan(calc_vector_blocks, (V, W), jnp.arange(0, blockSize * 2, 2))

    V_pairs = V.reshape(-1, 2, V.shape[1])
    W_pairs = W.reshape(-1, 2, W.shape[1])

    diffs = jnp.sum(V_pairs[:,0] * W_pairs[:,1] - V_pairs[:,1] * W_pairs[:,0], axis=1)
    result = jnp.prod(A[0::2, 1::2].diagonal() + diffs)

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

print('Pfaffian:', pfaffian(A))