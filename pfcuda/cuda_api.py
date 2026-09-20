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

_cuda_backend = None


def _require_cuda_backend():
    """Report a missing CUDA jaxlib instead of 'No FFI handler registered'.

    Probed on first use rather than at import, so that `import pfcuda` does not
    initialise a JAX backend as a side effect.
    """
    global _cuda_backend
    if _cuda_backend is None:
        try:
            _cuda_backend = bool(jax.devices('cuda'))
        except RuntimeError:
            _cuda_backend = False
    if not _cuda_backend:
        raise RuntimeError(
            'pfcuda GPU kernels need a CUDA-enabled jaxlib, but JAX reports '
            f'only {jax.devices()}. Install the plugin matching your driver:\n'
            '    pip install "pfcuda[cuda13]"   (or "pfcuda[cuda12]")'
        )


# Per input dtype: FFI handler suffix, and the width of log|Pf|, which mirrors
# ProjectionType<T> in include/pfaffian_utils.cuh.
DTYPES = {
    jnp.dtype(jnp.float32): ('_f32', jnp.float32),
    jnp.dtype(jnp.float64): ('_f64', jnp.float64),
    jnp.dtype(jnp.complex64): ('_c64', jnp.float32),
    jnp.dtype(jnp.complex128): ('_c128', jnp.float64)
}


def _dispatch(A):
    try:
        return DTYPES[A.dtype]
    except KeyError:
        raise TypeError(
            f'pfcuda does not support {A.dtype}; supported dtypes are '
            + ', '.join(str(d) for d in DTYPES)
        ) from None

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

    suffix, _ = _dispatch(A)
    _require_cuda_backend()

    func = jax.ffi.ffi_call(
        'pfaffian' + suffix,
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

    suffix, real_dtype = _dispatch(A)

    if n & 1:
        return (
            jnp.array(-jnp.inf, dtype=real_dtype),
            jnp.array(0, dtype=A.dtype)
        )

    _require_cuda_backend()

    func = jax.ffi.ffi_call(
        'slog_pfaffian' + suffix,
        (jax.ShapeDtypeStruct((), real_dtype), jax.ShapeDtypeStruct((), A.dtype)),
        input_layouts=[(1, 0)],
        vmap_method='broadcast_all'
    )

    return func(A)

def _dlog_pfaffian(A, A_dot):
    """d(log Pf) = tr(A^-1 @ A_dot) / 2, without forming the n x n product."""
    return 0.5 * jnp.einsum('ij,ji->', jnp.linalg.inv(A), A_dot)


@pfaffian.defjvp
def pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    primal_out = pfaffian(A)
    return primal_out, primal_out * _dlog_pfaffian(A, A_dot)


@slog_pfaffian.defjvp
def slog_pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    log_mag, sign = slog_pfaffian(A)

    # d(log Pf) splits into d(log|Pf|) = Re and d(arg Pf) = Im. The phase term
    # matters only for complex input, where sign is a unit complex number
    # rather than +-1; for real input Im vanishes.
    dlog = _dlog_pfaffian(A, A_dot)
    tangent_log_mag = jnp.real(dlog).astype(log_mag.dtype)

    if jnp.issubdtype(sign.dtype, jnp.complexfloating):
        tangent_sign = 1j * sign * jnp.imag(dlog)
    else:
        tangent_sign = jnp.zeros_like(sign)

    return (log_mag, sign), (tangent_log_mag, tangent_sign)