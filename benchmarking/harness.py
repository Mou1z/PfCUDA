"""Timing and accuracy measurement shared by every implementation.

Protocol, applied identically to all implementations so the comparison is fair:

* Each implementation receives the same host NumPy matrices, generated once per
  (size, sample) from a fixed seed.
* The timed region is the whole call, including host-to-device transfer and the
  synchronisation needed to make the result available. That is what a caller
  waits for, so dispatch overhead is included by design.
* One warm-up call per implementation and size is discarded, so neither CUDA
  context creation nor JIT compilation is counted.
* Latency is the median over the samples. Run-to-run variance on a loaded
  machine can exceed 4x, which a mean would present as measurement precision.

Accuracy is the disagreement between log|Pf| and 0.5*logabsdet from
numpy.linalg.slogdet, computed in float64. slogdet is an independent O(n^3)
factorisation rather than exact arithmetic, so its own roundoff is of the same
order as the values being compared; this measures agreement with a strong
reference, not absolute error.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Measurement:
    implementation: str
    method: str
    device: str
    n: int
    samples: int
    latency_ms: float | None
    latency_min_ms: float | None
    mean_log_error: float | None
    max_log_error: float | None
    skipped: str | None = None

    def as_dict(self):
        return asdict(self)


def skew_matrix(n, index, seed=0xB0A7):
    """One skew-symmetric matrix, reproducible per (size, index).

    Seeded per index rather than drawn from a running generator so matrices can
    be produced one at a time. Holding a whole sample set resident costs 2.7 GB
    at n=4096 with 20 samples, which is enough to exhaust a WSL VM.
    """
    rng = np.random.default_rng((seed, n, index))
    A = rng.normal(size=(n, n))
    return np.asarray(A - A.T, dtype=np.float64)


def reference_log_abs(A):
    """0.5 * log|det A| = log|Pf A| for skew-symmetric A."""
    return 0.5 * np.linalg.slogdet(A)[1]


def measure(impl, n, samples, budget_s=20.0, min_samples=3):
    """Time and score one implementation at one size.

    A slow implementation at a large size would otherwise dominate the run, so
    the warm-up call is used to estimate cost and the sample count is reduced
    to fit `budget_s`, never below `min_samples`. The count actually used is
    recorded alongside the result.
    """
    reason = impl.unsupported(n)
    if reason:
        return Measurement(
            impl.name, impl.method, impl.device, n, 0,
            None, None, None, None, skipped=reason,
        )

    try:
        warmup_start = time.perf_counter()
        impl.log_abs_pfaffian(skew_matrix(n, 0))
        warmup_s = time.perf_counter() - warmup_start
    except Exception as exc:
        return Measurement(
            impl.name, impl.method, impl.device, n, 0,
            None, None, None, None,
            skipped=f"{type(exc).__name__}: {exc}"[:120],
        )

    if warmup_s > budget_s:
        # Report the warm-up as the single measurement rather than spending
        # another `samples` calls on it. run_suite retires the implementation
        # so larger sizes do not repeat the cost.
        return Measurement(
            impl.name, impl.method, impl.device, n, 1,
            warmup_s * 1e3, warmup_s * 1e3, None, None,
            skipped=f"exceeded the {budget_s:g}s budget for one call",
        )

    if warmup_s > 0:
        samples = max(min_samples, min(samples, int(budget_s / warmup_s) or min_samples))

    timings = []
    errors = []
    for index in range(samples):
        A = skew_matrix(n, index)
        ref = reference_log_abs(A)
        start = time.perf_counter()
        value = impl.log_abs_pfaffian(A)
        timings.append((time.perf_counter() - start) * 1e3)
        errors.append(abs(float(value) - ref))
        del A

    return Measurement(
        implementation=impl.name,
        method=impl.method,
        device=impl.device,
        n=n,
        samples=samples,
        latency_ms=float(np.median(timings)),
        latency_min_ms=float(np.min(timings)),
        mean_log_error=float(np.mean(errors)),
        max_log_error=float(np.max(errors)),
    )
