import numpy as np

def pfaffian_py(A):
    n = A.shape[0]

    if (n == 0) or (n & 1) == 1:
        return 0.0
    elif n == 2:
        return A[0, 1]
    elif n == 4:
        return (
            A[0, 1] * A[2, 3] -
            A[0, 2] * A[1, 3] +
            A[0, 3] * A[1, 2]
        )

    result = 1.0

    V = A[0::2, :]
    W = A[1::2, :]

    def dR(V, W, k):
        return (
            V[:i, k] @ W[:i, k + 1:] - 
            W[:i, k] @ V[:i, k + 1:]
        )

    for i, k in enumerate(range(0, n - 1, 2)):
        Ak = A[k, k + 1:]
        Ak += dR(V, W, k)

        km = np.abs(Ak).argmax()
        
        if Ak[km] == 0:
            return 0.0

        if km != 0:
            kp = k + 1 + km

            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]

            result *= -1
        

        A[k + 1, k + 2:] = -(A[k + 1, k + 2:] + dR(V, W, k + 1))

        Ak[1:] /= Ak[0]
        result *= Ak[0]

    return result