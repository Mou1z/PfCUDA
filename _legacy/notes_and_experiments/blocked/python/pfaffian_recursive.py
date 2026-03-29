import numpy as np
from pfapack.pfaffian import pfaffian as pf

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=120,
    threshold=1000
)


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

memo = {}

def update_memoized(i, j):
    # Check if we've already done this work
    if (i, j) in memo:
        return memo[(i, j)]
    
    # Base Case
    if i < 2:
        return A[i, j]
    
    # Calculation
    res = (update_memoized(i - 1, i + 1) * update_memoized(i - 2, i - 1) - 
           update_memoized(i - 1, i) * update_memoized(i - 2, i + 1))
    
    # Store the result before returning
    memo[(i, j)] = res
    return res

update_memoized(8, 9)
print(memo)
n = A.shape[0]
for i in range(0, n - 1, 2):
    print(memo[(i, i + 1)])




print()
# print('Pfaffian:', pfaffian(A.copy()))
# print('Expected:', pf(A.copy()))