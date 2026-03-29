import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=120,
    threshold=1000
)

def pfaffian(A, blockSize = 4):
    result = 1.0
    n = A.shape[0]

    for i in range(0, n, blockSize):

        V = np.zeros((n, blockSize))
        W = np.zeros((n, blockSize))

        remRows = blockSize
        if i + remRows > n:
            remRows = n - i

        for j in range(0, remRows, 2): 
            k = i + j

            VWt = V @ W.T

            dVW = VWt - VWt.T
            dAk = A[k] + dVW[k]

            V[k + 2:, j] = dAk[k + 2:] / dAk[k + 1]
            W[:, j] = A[:, k + 1] + dVW[:, k + 1]

        VWt = V @ W.T
        fA = A + (VWt - VWt.T)

        A = fA

        for j in range(0, remRows, 2): 
            k = i + j

            print(fA[k, k + 1])

            result *= fA[k, k + 1]

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