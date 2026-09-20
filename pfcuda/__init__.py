from .cpp_api import pfaffian_cpu
from .pfaffian_py import pfaffian_py

# Imported lazily so the CPU backends stay usable without a CUDA runtime or a
# CUDA-enabled jaxlib. Calling a GPU function then reports the original cause.
try:
    from .cuda_api import pfaffian, slog_pfaffian

    CUDA_AVAILABLE = True
    CUDA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host machine
    CUDA_AVAILABLE = False
    CUDA_IMPORT_ERROR = exc

    def _gpu_unavailable(name):
        def stub(*args, **kwargs):
            raise RuntimeError(
                f"pfcuda.{name}() needs the CUDA backend, which failed to load:\n"
                f"    {type(CUDA_IMPORT_ERROR).__name__}: {CUDA_IMPORT_ERROR}\n"
                "This usually means no NVIDIA GPU, no CUDA runtime, or a jax "
                "build without CUDA support. Use pfaffian_cpu() or "
                "pfaffian_py() on CPU-only machines."
            ) from CUDA_IMPORT_ERROR

        stub.__name__ = name
        return stub

    pfaffian = _gpu_unavailable("pfaffian")
    slog_pfaffian = _gpu_unavailable("slog_pfaffian")

__all__ = [
    "pfaffian",
    "slog_pfaffian",
    "pfaffian_cpu",
    "pfaffian_py",
    "CUDA_AVAILABLE",
    "CUDA_IMPORT_ERROR",
]
