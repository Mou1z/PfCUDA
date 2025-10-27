# This is currently incomplete

import numpy as np

def optimalBlockSize(n, stride):
    semisqr = lambda x: x ** 1.2
    semicub = lambda x: x ** 1.455

    blockSize = stride

    while semicub(blockSize) - semisqr(blockSize) - (n/6) < 0.0:
        blockSize += stride

    return blockSize

def pfaffian(A):
    n = A.shape[0]

    blockSize = optimalBlockSize(n, 4)
    
    W = [0] * (blockSize * n)
    G = np.zeros_like(A)
    iPov = list(range(n))

    flipCount = 0

    for i in range(0, n-1, blockSize):
        currentBlockSize = (n - 2 - i) if (blockSize + i >= n - 2) else blockSize

        for j in range(currentBlockSize):
            currentIndex = i + j

            if j == 0:
                for k in range(currentIndex + 1, n):
                    G[k, i + j] = -A[k, i]
            else:
                G[currentIndex:, currentIndex] = W[currentIndex:, j - 1]

            for k in range(currentIndex + 1):
                G[k, currentIndex] = 0.0
            
            if abs(G[currentIndex + 1, currentIndex]) < 1e-3:
                ki = currentIndex + 1
                kp = np.argmax(G[currentIndex + 1:, currentIndex])

                if G[kp, currentIndex] > G[ki, currentIndex]:

                    flipCount += 1

                    iPov[ki], iPov[kp] = iPov[kp], iPov[ki]

                    if j != 0:
                        W[[ki, kp], :j] = W[[kp, ki], :j]
                    
                    if currentIndex != 0:
                        G[[ki, kp], :currentIndex] = G[[kp, ki], :currentIndex]
                    
                    A[:ki, [ki, kp]] = A[:ki, [kp, ki]]

                    A[ki, kp] *= -1.0

                    if kp - ki > 1:
                        A[ki+1:kp, kp], A[ki, ki+1:kp] = A[ki, ki+1:kp].copy(), A[ki+1:kp, kp].copy()

                    if kp + 1 < n:
                        A[ki, kp+1:], A[kp, kp+1:] = A[kp, kp+1:].copy(), A[ki, kp+1:].copy()
                    
                    G[ki, currentIndex], G[kp, currentIndex] = G[kp, currentIndex], G[ki, currentIndex]

            for k in range(currentIndex):
                W[k, j] = 0.0
            W[currentIndex, currentIndex] = A[currentIndex, currentIndex + 1]
            W[currentIndex + 1, currentIndex] = 0.0

            for k in range(currentIndex + 2, n):
                W[k, currentIndex] = -A[currentIndex + 1, k]
            
            alpha_k = G[currentIndex + 1, currentIndex]

            for k in range(currentIndex + 1, n):
                G[k, currentIndex] /= alpha_k
            
            G[currentIndex + 1, currentIndex] = 0.0

            A[currentIndex, currentIndex + 1] = - alpha_k

            if j != 0:

                vA[icur:] = -1.0 * Sp[icur:, :i] @ exG[icur+1:, 0] + vA[icur:]

                W[currentIndex:]