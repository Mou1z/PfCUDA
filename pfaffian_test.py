import math
import jax
import numpy as np
import jax.numpy as jnp
import lrux
from datetime import datetime
from pfaffian import pfaffian
# from pfcuda import pfaffian_f64 as pfaffian

from time import sleep

# from pfaffian import pfaffian, slog_pfaffian

def generate_skew_symmetric(n, key, scale=1.0):
    # Generate random n x n matrix
    mat = jax.random.normal(key, (n, n))
    
    # Make it skew-symmetric: A - A^T ensures A^T = -A
    skew_mat = mat - mat.T
    
    # Optional: zero diagonal (explicit)
    skew_mat = skew_mat - jnp.diag(jnp.diag(skew_mat))
    
    # Scale the values
    skew_mat = skew_mat * scale
    
    return skew_mat

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

print(pfaffian(matrix))