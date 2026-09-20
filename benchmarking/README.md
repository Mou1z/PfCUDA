# Benchmark methodology

```bash
./dev bench              # full sweep: writes JSON and the comparison figures
./dev bench --quick      # three sizes, five samples, no figures
./dev bench --compare    # per-size deltas against the stored baseline
./dev bench --accept     # promote this run to the baseline
```

The first run installs the competitor libraries listed in
`requirements-bench.txt`, several gigabytes because of PyTorch.

## What is measured

Wall-clock latency of one call, from a host NumPy matrix to a usable result,
including host-to-device transfer and the synchronisation that makes the value
available. Framework dispatch is therefore **included by design**: it is part of
what a caller waits for, and every implementation pays its own.

Two suites mirror the API split:

| Suite | Sizes | PfCUDA entry point |
| --- | --- | --- |
| `pfaffian` | 2–32 | `pfaffian()`, returns the value |
| `slog` | 34–4096 | `slog_pfaffian()`, returns `(log\|Pf\|, sign)` |

## Protocol

* Every implementation runs in **its own subprocess**. Loading JAX and PyTorch
  together distorts both: PfCUDA measured 38 ms at n=512 alone and 78 ms
  sharing a process. Isolation also bounds the damage from a hang, and each
  subprocess has a timeout.
* Everything runs in **float64**. JAX truncates float64 input to float32 unless
  x64 mode is enabled, which is faster and about nine orders of magnitude less
  accurate; the adapters refuse to run if it is not set, rather than quietly
  producing that comparison.
* Every implementation receives **byte-identical matrices**, generated one at a
  time from a per-index seed, so accuracy differences are real rather than
  sampling noise. Holding a full sample set resident costs 2.7 GB at n=4096.
* One **warm-up call per implementation and size** is discarded, so neither CUDA
  context creation nor JIT compilation is counted.
* Latency is the **median** over samples. Run-to-run variance on a loaded
  machine can exceed 4x; a mean would present that as measurement precision.
* The warm-up also estimates cost, and the sample count is reduced to keep each
  point within a 20 s budget, never below 3. The count actually used is recorded
  with every measurement.

## Accuracy

Error is `|log|Pf| - 0.5 * logabsdet|`, with `numpy.linalg.slogdet` in float64 as
the reference, averaged over the samples at each size.

`slogdet` is an independent LU factorisation, not exact arithmetic. Its own
roundoff is the same order as the values being compared, so **this measures
agreement with a strong independent reference, not absolute error**. Values at
the double-precision floor should be read as "indistinguishable from the
reference" rather than "exact".

## Keeping the comparison fair

Several competitors expose more than one algorithm. Leaving them on defaults
could understate them, so each choice is pinned deliberately and recorded in the
output JSON:

| Library | Choice | Why |
| --- | --- | --- |
| lrux | `householder` | Measured 2.31 ms vs 9.55 ms for `householder_for` at n=64 |
| TorchPfaffian | auto, `check_finite=False` | It selects a strategy from the input. Input validation is disabled because no other implementation validates, and it costs 48% at n=512 |

`pfapack` is included as the established CPU implementation, the tool this work
aims to improve on.

[PyPfaffian](https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian)
was evaluated and excluded. Its timings were not self-consistent: `log_pfaffian`
took 1.11 ms at n=64 where `pfaffian` took 16.15 ms for the same quantity, and
it then failed to return at n=128 despite an earlier n=256 measurement of
64 ms. Benchmarking against numbers that cannot be reproduced would not be
meaningful either way.

## Known limitations

* **pfapack has no log-domain entry point.** Its value overflows float64 once
  log\|Pf\| passes ~709, which for standard normal entries falls between n=400
  (3.2e246, finite) and n=512 (inf). It is capped at n=400 and guarded at
  runtime. This is a genuine limitation of value-returning APIs and the reason
  `slog_pfaffian` exists.
* **TorchPfaffian 0.0.5 imports `importlib_metadata` without declaring it**, so
  that dependency is listed explicitly in `requirements-bench.txt`.
* Results are hardware-specific. Every JSON records the GPU, CPU, library
  versions and git revision of the run that produced it.

## Output

| File | Contents |
| --- | --- |
| `benchmarks/<suite>_latest.json` | The most recent run |
| `benchmarks/<suite>_baseline.json` | The comparison point for `--compare` |
| `benchmarks/<suite>_comparison.png` | Latency and accuracy, side by side |

Each JSON is one self-contained run rather than an accumulating log, so any
published figure can be traced to a single identified configuration.
