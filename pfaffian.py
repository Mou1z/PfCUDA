import os
import jax
import ctypes
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', "cuda")

LIB_PATH = os.path.join(os.getcwd(), 'build', 'libcupfaffian.so')
lib = ctypes.cdll.LoadLibrary(LIB_PATH)

jax.ffi.register_ffi_target('pfaffian_f32', jax.ffi.pycapsule(lib.pfaffian_f32), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_f64', jax.ffi.pycapsule(lib.pfaffian_f64), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_c64', jax.ffi.pycapsule(lib.pfaffian_c64), platform='CUDA')
jax.ffi.register_ffi_target('pfaffian_c128', jax.ffi.pycapsule(lib.pfaffian_c128), platform='CUDA')

TYPE_MAPPINGS = {
    jnp.dtype(jnp.float32): '_f32',
    jnp.dtype(jnp.float64): '_f64',
    jnp.dtype(jnp.complex64): '_c64',
    jnp.dtype(jnp.complex128): '_c128'
}

@jax.jit
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

    if n <= 32:
        func = jax.ffi.ffi_call(
            'pfaffian' + TYPE_MAPPINGS[A.dtype],
            jax.ShapeDtypeStruct((), A.dtype),
            input_layouts=[(1, 0)],
            vmap_method='broadcast_all'
        )

    if n > 32:
        return 0

    return func(A)