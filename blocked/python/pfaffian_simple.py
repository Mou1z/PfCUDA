import numpy as np

def pfaffian(A):
    n = A.shape[0]
    if n & 1: return 0

    result = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.abs(A[k+1:, k]).argmax()

        if kp != k + 1:
            A[[k+1, kp], k:] = A[[kp, k+1], k:]
            A[:, [k+1, kp]] = A[:, [kp, k+1]]

            result *= -1

        if A[k, k+1] == 0.0:
            return 0.0

        result *= A[k, k+1]

        if k+2 < n:
            ak = A[k, k+2:] / A[k, k+1]
            sub = np.outer(ak, A[k+2:, k+1])

            A[k+2:, k+2:] += sub
            A[k+2:, k+2:] -= sub.T
        
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

print('Pfaffian:', pfaffian(A.copy()))