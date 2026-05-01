import os
import jax
import ctypes
import pathlib
import jax.numpy as jnp

LIB_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), 'libcupfaffian.so')
lib = ctypes.cdll.LoadLibrary(LIB_PATH)

jax.ffi.register_ffi_target('pfaffian_f32', jax.ffi.pycapsule(lib.pfaffian_f32), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_f64', jax.ffi.pycapsule(lib.pfaffian_f64), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_c64', jax.ffi.pycapsule(lib.pfaffian_c64), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_c128', jax.ffi.pycapsule(lib.pfaffian_c128), platform='CUDA')

jax.ffi.register_ffi_target('slog_pfaffian_f32', jax.ffi.pycapsule(lib.slog_pfaffian_f32), platform='CUDA')
jax.ffi.register_ffi_target('slog_pfaffian_f64', jax.ffi.pycapsule(lib.slog_pfaffian_f64), platform='CUDA')
jax.ffi.register_ffi_target('slog_pfaffian_c64', jax.ffi.pycapsule(lib.slog_pfaffian_c64), platform='CUDA')
jax.ffi.register_ffi_target('slog_pfaffian_c128', jax.ffi.pycapsule(lib.slog_pfaffian_c128), platform='CUDA')

TYPE_MAPPINGS = {
    jnp.dtype(jnp.float32): '_f32',
    jnp.dtype(jnp.float64): '_f64',
    jnp.dtype(jnp.complex64): '_c64',
    jnp.dtype(jnp.complex128): '_c128'
}

@jax.custom_jvp
def pfaffian(A):
    n = A.shape[0]

    if n == 0 or n & 1:
        return jnp.zeros((), dtype=A.dtype)
    
    if n == 2:
        return A[0, 1]
    
    if n == 4:
        return (
            A[0, 1] * A[2, 3] - 
            A[0, 2] * A[1, 3] + 
            A[0, 3] * A[1, 2]
        )

    if n > 32:
        raise ValueError("Matrix size exceeds the maximum supported size of 32x32.")

    func = jax.ffi.ffi_call(
        'pfaffian' + TYPE_MAPPINGS[A.dtype],
        jax.ShapeDtypeStruct((), A.dtype),
        input_layouts=[(1, 0)],
        vmap_method='broadcast_all'
    )

    return func(A)

@jax.custom_jvp
def slog_pfaffian(A):
    n = A.shape[0]

    if n <= 32:
        raise ValueError("Matrix size is less than the minimum supported size of (33x33). Use pfaffian() for smaller matrices.")

    if n & 1:
        return (
            jnp.array(-jnp.inf, dtype=jnp.float64),
            jnp.array(0, dtype=A.dtype)
        )

    func = jax.ffi.ffi_call(
        'slog_pfaffian' + TYPE_MAPPINGS[A.dtype],
        (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), A.dtype)),
        input_layouts=[(1, 0)],
        vmap_method='broadcast_all'
    )

    return func(A)

@pfaffian.defjvp
def pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents
    primal_out = pfaffian(A)

    A_inv = jnp.linalg.inv(A)

    product = jnp.matmul(A_inv, A_dot)
    product = primal_out * product

    tangent_out = 0.5 * jnp.linalg.trace(product)
    return primal_out, tangent_out


@slog_pfaffian.defjvp
def slog_pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents
    
    log_mag, sign = slog_pfaffian(A)
    primal_out = (log_mag, sign)
    
    A_inv = jnp.linalg.inv(A)
    
    tangent_log_mag = 0.5 * jnp.trace(jnp.matmul(A_inv, A_dot))
    tangent_sign = jnp.zeros_like(sign)
    
    tangent_out = (tangent_log_mag, tangent_sign)
    return primal_out, tangent_out