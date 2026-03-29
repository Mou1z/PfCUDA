import os
import ctypes
import jax
import jax.numpy as jnp
from jax.ffi import register_ffi_target, ffi_call, pycapsule

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda")

LIB_PATH = os.path.join(os.getcwd(), "build", "libcupfaffian.so")

register_ffi_target("pfaffian_f64", LIB_PATH, platform="cuda")

lib = ctypes.cdll.LoadLibrary(LIB_PATH)

# 3. Register the target using the CAPSULE, not the path
register_ffi_target("pfaffian_f64", pycapsule(lib.pfaffian_f64), platform="CUDA")

def cu_pfaffian_f64(matrix):
    
    # ffi_call returns a function. We call it immediately with (matrix)
    func = ffi_call(
        "pfaffian_f64",
        jax.ShapeDtypeStruct((), jnp.float64),
        input_layouts=[(1, 0)],
        vmap_method="broadcast_all"
    )

    return func(matrix)

matrix = jnp.array([
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    [-1,  0, 11, 12, 13, 14, 15, 16, 17, 18],
    [-2,-11,  0, 23, 24, 25, 26, 27, 28, 29],
    [-3,-12,-23,  0, 34, 35, 36, 37, 38, 39],
    [-4,-13,-24,-34,  0, 45, 46, 47, 48, 49],
    [-5,-14,-25,-35,-45,  0, 56, 57, 58, 59],
    [-6,-15,-26,-36,-46,-56,  0, 67, 68, 69],
    [-7,-16,-27,-37,-47,-57,-67,  0, 78, 79],
    [-8,-17,-28,-38,-48,-58,-68,-78,  0, 89],
    [-9,-18,-29,-39,-49,-59,-69,-79,-89,  0]
], dtype=jnp.float64)

# 4. Run the Test
try:
    print("--- Launching CUDA Pfaffian ---")
    result = cu_pfaffian_f64(matrix)
    
    # Trigger execution (JAX is lazy)
    print(f"Matrix Shape: {matrix.shape}")
    print(f"Pfaffian Result: {result.item()}")
    
    # Verification: For this specific matrix, the result should be 0 
    # (The Pfaffian of an NxN skew-symmetric matrix where N=10 and 
    # elements follow a simple linear pattern is often 0 due to linear dependence)
except Exception as e:
    print(f"Error during FFI call: {e}")