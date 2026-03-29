import jax
import jax.numpy as jnp
from jax.lax import cond, fori_loop
from jax import lax

@jax.custom_jvp
def pfaffian(A):
    n = A.shape[0]

    if n == 0 or n % 2 == 1:
        return 0.0

    def body(A):
        def iteration(j, args):
            A, result = args
            k = 2 * j

            indices = jnp.arange(n)
            mask = indices > k

            column = jnp.where(mask, jnp.abs(A[:, k]), -1.0)
            kp = jnp.argmax(column)

            def swap(args):
                A, result = args
                result *= -1.0

                row_perm = jnp.array([k+1, kp])
                A = A.at[row_perm, :].set(A[row_perm[::-1], :])
                A = A.at[:, row_perm].set(A[:, row_perm[::-1]])
                return A, result

            def step(args):
                A, result = args
                result *= A[k, k+1]

                def apply_tau(A):
                    mask2 = indices > (k+1)
                    tau = jnp.where(mask2, A[k, :], 0.0) / A[k, k+1]
                    col = jnp.where(mask2, A[:, k+1], 0.0)
                    update = jnp.outer(tau, col) - jnp.outer(col, tau)
                    A = A + update
                    return A

                A = cond(k + 2 < n, apply_tau, lambda A: A, A)
                return A, result

            A, result = cond(kp != k+1, swap, lambda args: args, (A, result))
            A, result = cond(A[k+1, k] != 0.0, step, lambda args: (args[0], 0.0), (A, result))
            return A, result

        _, result = fori_loop(0, n // 2, iteration, (A, 1.0))
        return result

    return body(A)

@pfaffian.defjvp
def pfaffian_jvp(primals, tangents):
    matrix, = primals
    direction, = tangents

    pf = pfaffian(matrix)

    size = matrix.shape[0]
    if size & 1:
        return 0, 0
    
    def fast_path(_):
        matrix_inverse = jnp.linalg.inv(matrix)
        trace = jnp.trace(matrix_inverse @ direction)
        return 0.5 * pf * trace

    def full_path(_):
        def get_minor(A, i, j):
            indices = jnp.array([k for k in range(size) if k != i and k != j])
            return A[jnp.ix_(indices, indices)]
        
        jvp = 0.0
        for i in range(size):
            for j in range(i + 1, size):
                sign = (-1) ** (i + j + 1)
                matrix_minor = get_minor(matrix, i, j)
                pf_minor = pfaffian(matrix_minor)
                jvp += sign * pf_minor * direction[i, j]
        return jvp
    
    det = jnp.linalg.det(matrix)
    jvp = lax.cond(
        jnp.abs(det) > 1e-12,
        fast_path,
        full_path,
        operand=None
    )

    return pf, jvp