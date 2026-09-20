"""Measure one implementation, in its own process, and emit JSON on stdout.

Each implementation is isolated because loading JAX and PyTorch together
distorts both: PfCUDA measured 38 ms at n=512 alone and 78 ms sharing a process
with torch. Isolation also means a hang or a crash costs one implementation
rather than the whole run.

Not intended to be run by hand; benchmarking/run.py spawns it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# JAX otherwise reserves 75% of the device on import, which matters as soon as
# anything else wants memory. Must precede the jax import inside the adapters.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Without this JAX silently truncates float64 input to float32, which is both
# faster and around nine orders of magnitude less accurate than the non-JAX
# implementations it is being compared against. Set through the environment so
# it applies however the adapters import jax.
os.environ.setdefault("JAX_ENABLE_X64", "true")

from benchmarking import implementations
from benchmarking.harness import measure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sizes", required=True, help="comma separated")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--budget", type=float, default=20.0)
    args = parser.parse_args(argv)

    try:
        impl = implementations.build(args.suite, args.name)
    except Exception as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"[:200]}, sys.stdout)
        return 0

    results = []
    for size in (int(s) for s in args.sizes.split(",")):
        m = measure(impl, size, args.samples, budget_s=args.budget)
        results.append(m.as_dict())
        # Streamed so the parent can show progress and so a later hang does not
        # discard what has already been measured.
        print(json.dumps({"result": m.as_dict()}), flush=True)
        if m.skipped and "budget" in m.skipped:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
