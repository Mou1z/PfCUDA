import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=120,
    threshold=1000
)

def pfaffian(A):
    result = 1.0
    n = A.shape[0]
    
    blockSize = n - 2
    
    V = np.zeros((n, blockSize))
    W = np.zeros((n, blockSize))

    for k in range(0, blockSize, 2):
        kp = k + 1 + np.abs(A[k, k + 1:]).argmax()

        if kp != k + 1:
            A[[k+1, kp], :] = A[[kp, k+1], :]
            A[:, [k+1, kp]] = A[:, [kp, k+1]]

            V[[k+1, kp], :] = V[[kp, k+1], :]
            W[[k+1, kp], :] = W[[kp, k+1], :]

            result *= -1

        uAk = A[k, k + 1:] + ((V[k] @ W.T[:, k + 1:]) - (W[k] @ V.T[:, k + 1:]))

        V[k + 2:, k] = uAk[1:] / uAk[0]
        W[:, k] = A[:, k + 1] + ((W[k + 1] @ V.T) - (V[k + 1] @ W.T))

    print(V)
    print(W)
    print(W.T)

    for k in range(0, n, 2):
        diff = (V[k] @ W[k + 1]) - (V[k + 1] @ W[k])
        result *= A[k, k + 1] + diff
    
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