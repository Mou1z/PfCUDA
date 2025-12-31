import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit
def pfaffian(A):
    # The Pfaffian squared is the determinant.
    # slogdet returns (sign, log_abs_determinant)
    sign, logdet = jnp.linalg.slogdet(A)
    
    # pf = sign * sqrt(exp(logdet)) -> sign * exp(0.5 * logdet)
    # We use the log-space to prevent overflow/underflow for 5000x5000
    return sign * jnp.exp(0.5 * logdet)

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