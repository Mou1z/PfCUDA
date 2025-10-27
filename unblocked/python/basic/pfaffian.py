import numpy as np

def pfaffian(A):
    n = A.shape[0]
    if n & 1: return 0

    result = 1.0

    for k in range(0, n-1, 2):
        kp = k + 1 + np.abs(A[k+1:, k]).argmax()

        if kp != k+1:
            A[[k+1, kp], k:] = A[[kp, k+1], k:]
            A[:, [k+1, kp]] = A[:, [kp, k+1]]

            result *= -1

        if A[k+1, k] != 0.0:
            result *= A[k, k+1]

            tau = A[k, k+2:].copy()
            tau /= A[k, k+1]

            if k+2<n:
                A[k+2:,k+2:] += np.outer(tau, A[k+2:,k+1])
                A[k+2:,k+2:] -= np.outer(A[k+2:,k+1], tau)
        else:
            return 0.0
        
    return result