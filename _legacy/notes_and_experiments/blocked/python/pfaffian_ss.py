import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=120,
    threshold=1000
)

def random_skew_symmetric_even(n, scale=1.0):
    if n % 2 != 0:
        raise ValueError("Matrix dimension n must be even.")

    M = scale * np.random.randn(n, n)
    A = M - M.T
    return A

# def pfaffian(A):
#     n = A.shape[0]
#     pf = 1.0

#     V = A[0::2, :]
#     W = A[1::2, :]

#     for k in range(n - 1):
#         Ak = A[k, k + 1:]

#         if k % 2 == 0:
#             kp = k + 1 + np.abs(Ak).argmax()

#             if A[k, kp] == 0:
#                 return 0.0

#             if kp != k + 1:
#                 A[[k + 1, kp], :] = A[[kp, k + 1], :]
#                 A[:, [k + 1, kp]] = A[:, [kp, k + 1]]

#                 pf *= -1

#             pf *= Ak[0]
#             Ak[1:] /= Ak[0]
#         else:
#             Ak *= -1

#         k += 1
#         i = k // 2

#         A[k, k + 1:] += (V[:i, k] @ W[:i, k + 1:] - W[:i, k] @ V[:i, k + 1:])

#     return pf

def pfaffian(A):
    n = A.shape[0]
    result = 1.0

    for i, k in enumerate(range(0, n - 1, 2)):
        V = A[0:k:2, :]
        W = A[1:k:2, :]

        rowUpdate = (V[:i, k] @ W[:i, k + 1:] - W[:i, k] @ V[:i, k + 1:])
        Ak = A[k, k + 1:]
        Ak += rowUpdate

        km = np.abs(Ak).argmax()
        kp = k + 1 + km

        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]

            result *= -1
        
        colUpdate = (V[:i, k + 1] @ W[:i, k + 2:] - W[:i, k + 1] @ V[:i, k + 2:])

        A[k + 1, k + 2:] = -(A[k + 1, k + 2:] + colUpdate)
        Ak[1:] /= Ak[0]
        
        result *= Ak[0]

    return result

import numpy as np

import numpy as np

def pfaffian_and_info(A_orig):
    n = A_orig.shape[0]
    A = A_orig.astype(float).copy()
    result = 1.0
    pivots = np.arange(n)
    # We will store BOTH L multipliers here
    L_factors = np.zeros((n, n))

    for i, k in enumerate(range(0, n - 1, 2)):
        V = A[0:k:2, :]
        W = A[1:k:2, :]

        # 1. Update row k
        rowUpdate = (V[:i, k] @ W[:i, k + 1:] - W[:i, k] @ V[:i, k + 1:])
        A[k, k + 1:] += rowUpdate

        # 2. Pivot
        km = np.abs(A[k, k + 1:]).argmax()
        kp = k + 1 + km

        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            pivots[[k + 1, kp]] = pivots[[kp, k + 1]]
            
            L_factors[[k + 1, kp], :] = L_factors[[kp, k + 1], :]
            result *= -1
        
        # 3. Update row k+1
        colUpdate = (V[:i, k + 1] @ W[:i, k + 2:] - W[:i, k + 1] @ V[:i, k + 2:])
        A[k + 1, k + 2:] = -(A[k + 1, k + 2:] + colUpdate)
        
        # 4. Extract Multipliers
        pivot_val = A[k, k + 1]
        result *= pivot_val
        if np.abs(pivot_val) > 1e-15:
            # FIX 1: Store BOTH multipliers in L_factors
            L_factors[k + 2:, k] = -A[k + 1, k + 2:] / pivot_val
            L_factors[k + 2:, k + 1] = A[k, k + 2:] / pivot_val
            
            # Store them in A as well for consistency
            A[k, k + 2:] /= pivot_val 
        else:
            return 0, None, None, None

    return result, A, pivots, L_factors

def get_inverse(pf, A_dec, pivots, L_factors):
    if pf == 0: return None
    n = A_dec.shape[0]
    inv = np.eye(n)

    # Step 1: Forward Substitution (L)
    for k in range(0, n - 2, 2):
        for j in range(k + 2, n):
            # FIX 2: Apply both columns of the L block
            inv[j, :] -= L_factors[j, k] * inv[k, :] + L_factors[j, k + 1] * inv[k + 1, :]

    # Step 2: Tridiagonal Solve (T) (Your code here was perfect)
    for k in range(0, n, 2):
        val = A_dec[k, k + 1]
        row_k = inv[k, :].copy()
        inv[k, :] = -inv[k + 1, :] / val
        inv[k + 1, :] = row_k / val

    # Step 3: Backward Substitution (L^T)
    for k in range(n - 4, -1, -2):
        for j in range(k + 2, n):
            # FIX 3: Apply both columns of the L^T block
            inv[k, :] -= L_factors[j, k] * inv[j, :]
            inv[k + 1, :] -= L_factors[j, k + 1] * inv[j, :]

    # FIX 4: Final Permutation Reversal
    # We use numpy's advanced indexing to invert the full permutation matrix at once
    final_inv = np.zeros_like(inv)
    final_inv[np.ix_(pivots, pivots)] = inv
    
    return final_inv

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

# # A = random_skew_symmetric_even(10, 0.1)

# print()
# print('Pfaffian:', pfaffian(A.copy()))
result, ret_A, pivots, L_factors = pfaffian_and_info(A.copy())
if np.abs(result) > 1e-15:
    # 3. Calculate the inverse using the gathered info
    my_inv = get_inverse(result, ret_A, pivots, L_factors)
    
    print("--- Pfaffian ---")
    print(result)
    
    print("\n--- My Inverse ---")
    print(my_inv)
    
    print("\n--- Validation (A @ A_inv should be Identity) ---")
    identity_check = A @ my_inv
    print(np.round(identity_check, 2)) 
else:
    print("Matrix is singular, no inverse exists.")