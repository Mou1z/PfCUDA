"""Adapters presenting each library through one uniform interface.

Every adapter takes a host NumPy matrix and returns log|Pf| as a float, with
device work synchronised before returning, so `harness.measure` times
comparable work in every case.

Where a library offers several algorithms the choice is pinned explicitly
rather than left to its default, and recorded in the results, so a reader can
see which variant was measured. See METHOD_NOTES.
"""

from __future__ import annotations

import math

import numpy as np

# Fastest correct variant per library, measured on this project's benchmark
# machine; see benchmarking/README.md for the survey that selected them.
METHOD_NOTES = {
    "lrux": "householder",
    # TorchPfaffian picks its own strategy from the input. check_finite is
    # disabled because no other implementation validates its input, and the
    # comparison should not charge one library for a safety check.
    "TorchPfaffian": "auto",
}

# pfapack offers no log-domain entry point, so it returns an unscaled value that
# overflows float64 once log|Pf| passes ~709. For standard normal entries that
# happens between n=400 (3.2e246, finite) and n=512 (inf), so cap below it and
# guard at runtime in case a different matrix scale overflows sooner.
PFAPACK_MAX_N = 400


def require_jax_x64(jnp):
    """Refuse to benchmark JAX in single precision.

    JAX truncates float64 input to float32 unless x64 mode is enabled, which is
    both faster and far less accurate than the non-JAX implementations it is
    compared against. Failing loudly beats publishing that comparison.
    """
    if jnp.zeros(1, dtype=jnp.float64).dtype != np.float64:
        raise RuntimeError("jax is in float32 mode; set JAX_ENABLE_X64=true")


class Implementation:
    """One library, measured through one entry point."""

    name = "?"
    method = "-"
    device = "cpu"
    min_n = 2
    max_n = None

    def unsupported(self, n):
        if n < self.min_n:
            return f"needs n >= {self.min_n}"
        if self.max_n is not None and n > self.max_n:
            return f"needs n <= {self.max_n}"
        return None

    def log_abs_pfaffian(self, A):
        raise NotImplementedError


# --- PfCUDA ----------------------------------------------------------------


class PfCUDAValue(Implementation):
    name = "PfCUDA"
    method = "pfaffian"
    device = "cuda"
    max_n = 32

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import pfcuda

        self._jax, self._jnp, self._pfcuda = jax, jnp, pfcuda
        require_jax_x64(jnp)

    def log_abs_pfaffian(self, A):
        value = self._pfcuda.pfaffian(A)
        self._jax.block_until_ready(value)
        return math.log(abs(float(value)))


class PfCUDASlog(Implementation):
    name = "PfCUDA"
    method = "slog_pfaffian"
    device = "cuda"
    min_n = 34

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import pfcuda

        self._jax, self._jnp, self._pfcuda = jax, jnp, pfcuda
        require_jax_x64(jnp)

    def log_abs_pfaffian(self, A):
        log_abs, sign = self._pfcuda.slog_pfaffian(A)
        self._jax.block_until_ready((log_abs, sign))
        return float(log_abs)


# --- lrux ------------------------------------------------------------------


class LruxValue(Implementation):
    name = "lrux"
    method = METHOD_NOTES["lrux"]
    device = "cuda"
    max_n = 32

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import lrux

        self._jax, self._jnp, self._lrux = jax, jnp, lrux
        require_jax_x64(jnp)

    def log_abs_pfaffian(self, A):
        value = self._lrux.pf(A, method=self.method)
        self._jax.block_until_ready(value)
        return math.log(abs(float(value)))


class LruxSlog(Implementation):
    name = "lrux"
    method = METHOD_NOTES["lrux"]
    device = "cuda"
    min_n = 34

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import lrux

        self._jax, self._jnp, self._lrux = jax, jnp, lrux
        require_jax_x64(jnp)

    def log_abs_pfaffian(self, A):
        result = self._lrux.slogpf(A, method=self.method)
        self._jax.block_until_ready(result)
        return float(result[1])


# --- TorchPfaffian ---------------------------------------------------------


class _TorchImplementation(Implementation):
    device = "cuda"

    def _to_device(self, A):
        return self._torch.as_tensor(A, dtype=self._torch.float64, device="cuda")

    def _sync(self):
        self._torch.cuda.synchronize()


class TorchPfaffianValue(_TorchImplementation):
    name = "TorchPfaffian"
    method = METHOD_NOTES["TorchPfaffian"]
    max_n = 32

    def __init__(self):
        import torch
        import torch_pfaffian

        self._torch, self._lib = torch, torch_pfaffian

    def log_abs_pfaffian(self, A):
        value = self._lib.pfaffian(self._to_device(A), check_finite=False)
        self._sync()
        return math.log(abs(float(value)))


class TorchPfaffianSlog(_TorchImplementation):
    name = "TorchPfaffian"
    method = METHOD_NOTES["TorchPfaffian"]
    min_n = 34

    def __init__(self):
        import torch
        import torch_pfaffian

        self._torch, self._lib = torch, torch_pfaffian

    def log_abs_pfaffian(self, A):
        _, log_abs = self._lib.slog_pfaffian(self._to_device(A))
        self._sync()
        return float(log_abs)


# --- pfapack, the established CPU implementation ----------------------------


class Pfapack(Implementation):
    name = "pfapack"
    method = "LTL"
    device = "cpu"
    max_n = PFAPACK_MAX_N

    def __init__(self):
        from pfapack.pfaffian import pfaffian

        self._pfaffian = pfaffian

    def log_abs_pfaffian(self, A):
        # pfapack factors in place, so hand it a copy.
        value = float(self._pfaffian(A.copy()))
        if not math.isfinite(value):
            raise OverflowError("Pfaffian exceeds float64 range; no log-domain API")
        return math.log(abs(value))


VALUE_SUITE = [PfCUDAValue, LruxValue, TorchPfaffianValue, Pfapack]
SLOG_SUITE = [PfCUDASlog, LruxSlog, TorchPfaffianSlog, Pfapack]


def suite_classes(suite):
    return VALUE_SUITE if suite == "pfaffian" else SLOG_SUITE


def names(suite):
    return [cls.name for cls in suite_classes(suite)]


def build(suite, name):
    """Instantiate one implementation by name, raising if it cannot load."""
    for cls in suite_classes(suite):
        if cls.name == name:
            return cls()
    raise KeyError(f"no implementation named {name!r} in suite {suite!r}")
