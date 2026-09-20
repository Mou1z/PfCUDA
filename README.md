# PfCUDA

[![CI](https://github.com/Mou1z/PfCUDA/actions/workflows/ci.yml/badge.svg)](https://github.com/Mou1z/PfCUDA/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pfcuda.svg)](https://pypi.org/project/pfcuda/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GPU-accelerated Pfaffian computation for JAX.**

Every skew-symmetric matrix `A` has a Pfaffian: a number whose square is the
determinant, `Pf(A)^2 = det(A)`. Unlike the determinant it keeps a meaningful
sign, which is why it turns up in quantum many-body physics.

PfCUDA computes it with CUDA kernels that plug into JAX, so the result works
with `jax.grad`, `jax.jit` and `jax.vmap` like any other JAX operation. C++ and
NumPy versions are included for machines without a GPU.

---

## ⚙️ Requirements

Installing PfCUDA compiles CUDA code on your machine, so you need the CUDA
Toolkit — a graphics driver on its own is not enough.

| Requirement | Notes |
| --- | --- |
| NVIDIA GPU | Compute capability 7.5 or newer if you build with CUDA 13, which dropped support for older cards. Older CUDA versions still support older GPUs. |
| CUDA Toolkit | Supplies `nvcc`. Found automatically on `PATH`, in `CUDA_HOME`, or at `/usr/local/cuda`. |
| CMake ≥ 3.18 and a C++17 compiler | |
| Python ≥ 3.10 | Including the development headers (`python3-dev` on Debian/Ubuntu). |
| `jax` ≥ 0.5.0 | Earlier versions lack the interface PfCUDA plugs into. Tested against 0.11.1. |

**Platform support** follows JAX's own CUDA support:

| Platform | Status |
| --- | --- |
| Linux x86_64 / aarch64 | Supported |
| Windows via WSL2 | Works; JAX calls WSL2 CUDA support experimental |
| Native Windows | **Not supported** — JAX has no CUDA wheels for it |
| macOS | **Not supported** — no NVIDIA CUDA |

On native Windows, install [WSL2](https://learn.microsoft.com/windows/wsl/install)
and use PfCUDA inside it. The CPU backends (`pfaffian_cpu`, `pfaffian_py`) work
anywhere the package can be built.

---

## 📦 Installation

Pick the extra matching your driver's CUDA version (`nvidia-smi` reports it):

```bash
pip install "pfcuda[cuda13]"    # or "pfcuda[cuda12]"
```

This compiles on your machine and takes a couple of minutes.

**Don't skip the `[cuda13]` part.** Plain `pip install pfcuda` installs the
CPU-only build of JAX. Everything compiles, and then the GPU functions fail
when you call them.

If `nvcc` is somewhere unusual, say where:

```bash
CUDA_HOME=/path/to/cuda pip install "pfcuda[cuda13]"
```

**Building on a machine with no GPU works**, which matters in a container, in
CI, or on a cluster login node where you build before submitting to a GPU node.
PfCUDA compiles for every GPU generation your CUDA Toolkit supports, so the
result runs wherever you send it. If you only care about the card in the build
machine, this is smaller and about twice as fast to compile:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native" pip install "pfcuda[cuda13]"
```

To install from a clone instead:

```bash
git clone https://github.com/Mou1z/PfCUDA.git
cd PfCUDA
pip install .
```

> **Upgrading JAX later?** Reinstall PfCUDA afterwards. The compiled kernels
> are tied to the exact JAX version they were built against, and JAX does not
> promise that interface stays compatible between releases.
>
> ```bash
> pip install --force-reinstall --no-binary pfcuda pfcuda
> ```

---

## 🚀 Quick Start

```python
import numpy as np
import jax, jax.numpy as jnp
import pfcuda

jax.config.update("jax_enable_x64", True)

A = np.array([
    [ 0.0,  1.0,  2.0,  3.0],
    [-1.0,  0.0,  4.0,  5.0],
    [-2.0, -4.0,  0.0,  6.0],
    [-3.0, -5.0, -6.0,  0.0],
], dtype=np.float64)

pfcuda.pfaffian(jnp.array(A))      # GPU, matrices up to 32x32  -> 8.0
pfcuda.pfaffian_cpu(A.copy())      # C++ backend, any even size -> 8.0
pfcuda.pfaffian_py(A.copy())       # NumPy reference            -> 8.0

# slog_pfaffian is for large matrices and requires n >= 34.
rng = np.random.default_rng(0)
B = rng.normal(size=(64, 64))
B = B - B.T
log_abs, sign = pfcuda.slog_pfaffian(jnp.array(B))
```

`pfcuda.CUDA_AVAILABLE` reports whether the GPU backend loaded; the CPU
functions remain usable when it did not.

---

## 📚 API Reference

Every input must be **square, skew-symmetric (`A == -A.T`), and have an even
number of rows**. The Pfaffian of an odd-sized matrix is zero, and each
function says so differently: `pfaffian` returns `0`, `slog_pfaffian` returns
`(-inf, 0)`, `pfaffian_py` returns `0.0`, and `pfaffian_cpu` raises
`RuntimeError`.

**Which one should you use?** `pfaffian` for matrices up to 32×32 on a GPU,
`slog_pfaffian` for anything larger, and the CPU functions when there is no GPU
or the matrix is small enough that a GPU is not worth it.

`slog_pfaffian` returns the logarithm rather than the value for a concrete
reason: past roughly n=500 the Pfaffian is too large for a double to hold, so
returning it directly would give you infinity.

| Function | Backend | Dtypes | Size | Returns |
| --- | --- | --- | --- | --- |
| `pfaffian(A)` | GPU (CUDA/JAX) | `float32`, `float64`, `complex64`, `complex128` | `n ≤ 32` | `Pf(A)` |
| `slog_pfaffian(A)` | GPU (CUDA/JAX) | `float32`, `float64`, `complex64`, `complex128` | `n ≥ 34` | `(log|Pf|, sign)` |
| `pfaffian_cpu(A)` | CPU (C++) | `float64` only | any even | `Pf(A)` |
| `pfaffian_py(A)` | CPU (NumPy) | any NumPy float dtype | any even | `Pf(A)` |

Two traps worth knowing about:

- **`pfaffian_cpu` only does float64.** Anything else gets converted on the way
  in, so a complex matrix silently loses its imaginary part and you get a wrong
  answer with no error. Use `pfaffian` for complex matrices.
- **The two CPU functions destroy the matrix you pass them** when `n > 4`; they
  factor it in place. Pass `A.copy()` if you still need it afterwards.

Both GPU functions are differentiable, so `jax.grad`, `jax.jit` and `jax.vmap`
work on them.

---

## 📊 Benchmarks

Against [lrux](https://pypi.org/project/lrux/) (JAX),
[TorchPfaffian](https://github.com/MatchCake/TorchPfaffian) (PyTorch) and
[pfapack](https://pypi.org/project/pfapack/) (the established CPU
implementation). All float64, on an NVIDIA GTX 1660 SUPER.

Timings are the **median milliseconds one call takes**, measured the way you
would actually use it: hand the function a NumPy matrix and wait for the
answer. That includes the overhead JAX and PyTorch add on top of the maths,
because you pay for that too.

Accuracy is how far each result sits from `0.5 × logabsdet`, averaged over 20
different matrices at each size. [benchmarking/README.md](benchmarking/README.md)
explains the full method.

![pfaffian() benchmark](benchmarks/pfaffian_comparison.png)
![slog_pfaffian() benchmark](benchmarks/slog_pfaffian_comparison.png)

**Large matrices — `slog_pfaffian`.** lrux is quicker up to around n=1024.
Past that point PfCUDA pulls ahead, and the gap widens fast:

| n | PfCUDA | lrux | TorchPfaffian |
| --- | --- | --- | --- |
| 1024 | 113 ms | 135 ms | 512 ms |
| 2048 | **215 ms** | 911 ms | 842 ms |
| 4096 | **644 ms** | 7245 ms | 4509 ms |

At n=4096 PfCUDA is 11.3× faster than lrux and 7.0× faster than TorchPfaffian.

**Small matrices — `pfaffian`, n = 2…32.** PfCUDA takes 0.17 ms up to n=4,
where it uses a direct formula, and 2.3–4.1 ms above that. lrux takes
1.8–7.6 ms and TorchPfaffian 0.3–13.5 ms.

At these sizes the actual arithmetic is trivial; almost all the time goes on
getting the work to the GPU and back. **pfapack on the CPU beats every GPU
implementation here** at 0.03–0.43 ms. A GPU only starts paying off around
n=256.

**Accuracy.** All four land within 10⁻¹⁴ to 10⁻¹¹ of the reference at every
size. Nobody is buying speed by cutting precision.

---

## 🛠️ Implementation

- **GPU** — CUDA kernels behind JAX's FFI with custom JVP rules.
  `src/pfaffian.cu`, `src/pfaffian_sm.cu`, `src/slog_pfaffian.cu`,
  `src/slog_pfaffian_lg.cu`, `bindings/jax_bindings.cu`
- **CPU (C++)** — pybind11 module. `src/pfaffian_cpu.cpp`,
  `bindings/pybind_bindings.cpp`
- **CPU (NumPy)** — pure-Python reference. `pfcuda/pfaffian_py.py`

---

## 🧑‍💻 Development

If you are changing the code, use `./dev` instead of `pip install .`. It
compiles straight into `pfcuda/`, so rebuilding after an edit takes about five
seconds rather than two minutes.

```bash
git clone https://github.com/Mou1z/PfCUDA.git
cd PfCUDA
./dev
```

The first run sets everything up: it creates `.venv`, works out whether your
driver needs the CUDA 12 or CUDA 13 build of JAX, installs the dependencies and
configures CMake. Later runs just rebuild what changed.

| Command | What it does |
| --- | --- |
| `./dev` | Rebuild and run the fast tests (~17 s) |
| `./dev build` | Rebuild only |
| `./dev test` | Rebuild and run the whole suite (~2 min) |
| `./dev bench` | Run the benchmarks (`--quick`, `--compare`) |
| `./dev doctor` | Report on your environment; changes nothing |
| `./dev clean` | Delete build outputs (`--all` also deletes `.venv`) |

On Windows, run `.\dev.ps1 <command>` from PowerShell and it forwards into
WSL2. If a build fails, `./dev doctor` usually shows why. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## 📝 Citation

Machine-readable metadata is in [CITATION.cff](CITATION.cff). To cite the work
behind it:

> **Muhammad Mouiz Ghouri**, *"Optimized Pfaffian Computation and Its
> Differentiation: From CPU Implementations to GPU Acceleration"*,
> Eötvös Loránd University, Budapest, Hungary, 2026.

## 🤝 Contributing

Issues and pull requests are welcome — especially wider dtype support on the
CPU backend, more GPU architectures, and further benchmark comparisons. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up, and please include
`./dev doctor` output in bug reports.

## 📄 License

[MIT](LICENSE)
