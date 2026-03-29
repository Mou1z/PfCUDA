import numpy as np
import pfaffian_module
from pfapack.pfaffian import pfaffian as pf

def random_skew_symmetric_even(n, scale=1.0):
    if n % 2 != 0:
        raise ValueError("Matrix dimension n must be even.")

    M = scale * np.random.randn(n, n)
    A = M - M.T
    return A

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

A = random_skew_symmetric_even(100, 0.1)

print()
print('Pfaffian:', pfaffian_module.pfaffian(A.copy()))
print('Expected:', pf(A.copy()))
print('Expected (Det-Based): ', np.sqrt(np.linalg.det(A)))