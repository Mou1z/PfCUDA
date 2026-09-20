"""Benchmark runner.

    python -m benchmarking.run                      full sweep, writes JSON and plots
    python -m benchmarking.run --quick              three sizes, few samples, no plot
    python -m benchmarking.run --compare            against the stored baseline
    python -m benchmarking.run --accept             promote the latest run to baseline
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys

# JAX reserves 75% of the device by default, which leaves a 6 GB card with too
# little for PyTorch when both competitors run in one process. Must be set
# before anything imports jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarking import implementations

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks"

SUITES = {
    # pfaffian() is limited to 32x32; slog_pfaffian() starts at 34x34.
    "pfaffian": {
        "sizes": list(range(2, 33, 2)),
        "quick_sizes": [4, 16, 32],
        "title": "pfaffian()",
    },
    "slog_pfaffian": {
        "sizes": [34, 64, 128, 256, 512, 1024, 2048, 3072, 4096],
        "quick_sizes": [64, 512, 2048],
        "title": "slog_pfaffian()",
    },
}


def _command(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# Import name -> distribution name, which differ for the two PyTorch libraries.
DISTRIBUTIONS = {
    "pfcuda": "pfcuda",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "lrux": "lrux",
    "torch": "torch",
    "torch_pfaffian": "torchpfaffian",
    "torchpfaff": "torchpfaff",
    "pfapack": "pfapack",
    "numpy": "numpy",
}


def environment():
    import importlib.metadata as md

    versions = {}
    for module, distribution in DISTRIBUTIONS.items():
        try:
            versions[module] = md.version(distribution)
        except Exception:
            versions[module] = "unknown"

    if versions["pfcuda"] == "unknown":
        # ./dev puts the repo on sys.path rather than installing it, so there
        # is no distribution metadata to read. The source is the version.
        import re

        text = (ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match:
            versions["pfcuda"] = f"{match.group(1)} (source)"

    gpu = _command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_rev": _command("git", "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_command("git", "status", "--porcelain")),
        "gpu": gpu,
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
    }


def run_implementation(suite, name, sizes, samples, timeout_s):
    """Measure one implementation in a subprocess, streaming its results."""
    command = [
        sys.executable, "-u", "-m", "benchmarking.worker",
        "--suite", suite, "--name", name,
        "--sizes", ",".join(str(s) for s in sizes),
        "--samples", str(samples),
    ]
    results = []
    try:
        process = subprocess.run(command, cwd=ROOT, text=True, timeout=timeout_s,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as expired:
        for line in (expired.stdout or "").splitlines():
            _collect(line, results)
        print(f"  {name}: timed out after {timeout_s:g}s, keeping "
              f"{len(results)} completed size(s)", file=sys.stderr)
        return results

    for line in process.stdout.splitlines():
        _collect(line, results)
    return results


def _collect(line, results):
    try:
        payload = json.loads(line)
    except ValueError:
        return
    if "result" in payload:
        results.append(payload["result"])
    elif "error" in payload:
        print(f"  unavailable: {payload['error']}", file=sys.stderr)


def run_suite(suite, sizes, samples, timeout_s):
    print(f"{'implementation':>16}  " + "  ".join(f"{n:>12}" for n in sizes),
          flush=True)
    results = []
    for name in implementations.names(suite):
        measured = run_implementation(suite, name, sizes, samples, timeout_s)
        results.extend(measured)

        by_size = {r["n"]: r for r in measured}
        cells = []
        for n in sizes:
            r = by_size.get(n)
            cells.append("-" if not r or r["latency_ms"] is None
                         else f"{r['latency_ms']:.3f}")
        print(f"{name:>16}  " + "  ".join(f"{c:>12}" for c in cells), flush=True)

    return results


def write(payload, path):
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {path.relative_to(ROOT)}")


def compare(current, baseline_path):
    if not baseline_path.exists():
        sys.exit(f"no baseline at {baseline_path.relative_to(ROOT)}; run --accept first")

    baseline = json.loads(baseline_path.read_text())
    previous = {
        (r["implementation"], r["method"], r["n"]): r["latency_ms"]
        for r in baseline["results"]
    }

    print(f"\nagainst baseline {baseline['environment']['git_rev']} "
          f"({baseline['environment']['timestamp']})")
    print(f"{'implementation':>16} {'n':>6} {'baseline':>11} {'current':>11} {'change':>9}")

    regressions = 0
    for r in current["results"]:
        key = (r["implementation"], r["method"], r["n"])
        before, after = previous.get(key), r["latency_ms"]
        if before is None or after is None:
            continue
        delta = (after - before) / before * 100
        if delta > 5:
            regressions += 1
        print(f"{r['implementation']:>16} {r['n']:>6} {before:>11.3f} {after:>11.3f} "
              f"{delta:>+8.1f}%")

    if regressions:
        print(f"\n{regressions} measurement(s) more than 5% slower than baseline")
    return regressions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("pfaffian", "slog_pfaffian", "both"), default="both")
    parser.add_argument("--samples", type=int, default=20,
                        help="distinct matrices per size (default 20)")
    parser.add_argument("--quick", action="store_true",
                        help="three sizes, 5 samples, no plot")
    parser.add_argument("--compare", action="store_true",
                        help="print deltas against the stored baseline")
    parser.add_argument("--accept", action="store_true",
                        help="promote the results of this run to the baseline")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds allowed per implementation (default 900)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    suites = ("pfaffian", "slog_pfaffian") if args.suite == "both" else (args.suite,)
    samples = 5 if args.quick else args.samples
    env = environment()
    if env["git_dirty"]:
        print("note: working tree has uncommitted changes\n", file=sys.stderr)

    exit_code = 0
    for suite in suites:
        spec = SUITES[suite]
        sizes = spec["quick_sizes"] if args.quick else spec["sizes"]
        print(f"\n=== {spec['title']}, {samples} matrices per size, "
              f"median latency in ms ===")

        payload = {
            "schema": 1,
            "suite": suite,
            "samples": samples,
            "environment": env,
            "results": run_suite(suite, sizes, samples, args.timeout),
        }

        if args.quick:
            continue

        write(payload, RESULTS_DIR / f"{suite}_latest.json")

        if args.compare:
            exit_code |= bool(compare(payload, RESULTS_DIR / f"{suite}_baseline.json"))
        if args.accept:
            write(payload, RESULTS_DIR / f"{suite}_baseline.json")
        if not args.no_plot:
            from benchmarking import plot

            plot.render(payload, RESULTS_DIR / f"{suite}_comparison.png")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
