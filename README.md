# PfCUDA

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Supported](https://img.shields.io/badge/CUDA-Supported-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pfaffian computation accelerated for GPU and CPU.**

---

## 📖 Overview

In mathematics and computational physics, computing the Pfaffian `Pf(A)` of a `2n x 2n` skew-symmetric matrix `A` (where `Pf(A)^2 = det(A)`) is a computationally demanding task. 

**PfCUDA** is a high-performance Pfaffian library designed to eliminate this bottleneck by combining:

* **CUDA + JAX FFI** for massive GPU-accelerated Pfaffian evaluation.
* **C++ / pybind11** for fast native CPU execution.
* **NumPy Python fallback** for portability, rapid prototyping, and validation.

The library makes Pfaffian calculations trivial from Python while exposing three distinct implementation layers tailored to different workloads, matrix sizes, and deployment environments.

## ✨ Key Features

* **GPU Acceleration:** Utilizes JAX's CUDA foreign function interface (FFI).
* **JVP Support:** Supports JAX Jacobian-vector products (JVP) for `pfaffian()` and `slog_pfaffian()`.
* **Native CPU Speeds:** Fast fallback using C++ with pybind11.
* **Reliable References:** Pure Python implementation using NumPy.
* **Broad Type Support:** Natively supports `float32`, `float64`, `complex64`, and `complex128` for GPU/C++ calculations, and virtually any dtype via NumPy.
* **High Performance:** Up to **~6.7x speedup** against existing JAX-based libraries (like Lrux) for large matrices.
* **Targeted:** Optimized specifically for skew-symmetric matrices and even matrix dimensions.

---

## ⚙️ Installation

To install PfCUDA, clone the repository and install it via `pip` from the root directory:

```bash
git clone https://github.com/Mou1z/PfCUDA.git
cd PfCUDA
pip install .

```

> **Note:** You must have a CUDA-enabled GPU and a matching CUDA Toolkit installed to utilize the `jax` and CUDA-backed functions.

---

## 🚀 Quick Start

Here is a minimal working example demonstrating the different backends:

```python
import numpy as np
import jax.numpy as jnp
import pfcuda

# Example 4x4 skew-symmetric matrix (complex types also supported!)
A = np.array([
    [ 0.0,  1.0,  2.0,  3.0],
    [-1.0,  0.0,  4.0,  5.0],
    [-2.0, -4.0,  0.0,  6.0],
    [-3.0, -5.0, -6.0,  0.0],
], dtype=np.float64)

# 1. GPU-backed JAX interface
A_jax = jnp.array(A)
pf = pfcuda.pfaffian(A_jax)

# 2. Signed log Pfaffian for larger matrices (GPU)
log_abs, sign = pfcuda.slog_pfaffian(A_jax)

# 3. CPU implementation via C++
pf_cpu = pfcuda.pfaffian_cpu(A)

# 4. NumPy pure Python fallback
pf_py = pfcuda.pfaffian_py(A)

print(f"Pfaffian (GPU): {pf}")

```

---

## 📚 API Reference

`pfcuda` requires all input matrices to be **square, skew-symmetric, and of even dimensions**.

| Function | Backend | Supported Dtypes | Max Size | Return Value |
| --- | --- | --- | --- | --- |
| `pfcuda.pfaffian(A)` | GPU (CUDA/JAX) | `float32`, `float64`, `complex64`, `complex128` | `32 x 32` | Exact value |
| `pfcuda.slog_pfaffian(A)` | GPU (CUDA/JAX) | `float32`, `float64`, `complex64`, `complex128` | Unlimited (Even) | `(log_abs, sign)` |
| `pfcuda.pfaffian_cpu(A)` | CPU (C++) | Core C++ supported types | Unlimited (Even) | Exact value |
| `pfcuda.pfaffian_py(A)` | CPU (NumPy) | Any compatible NumPy dtype | Unlimited (Even) | Exact value |

---

## 📊 Benchmarks vs. Lrux

Benchmark scripts are provided in the `benchmarking/` directory (`benchmark_pfcuda.py` and `benchmark_pfcuda_sm.py`). These scripts compare PfCUDA's performance against **Lrux**, an existing JAX-based library.


![PfCUDA vs Lrux slog_pfaffian benchmark comparison](benchmarking/slog_pfaffian_comparison.png)
*(See `benchmarks/pfaffian_data.json` for raw benchmark data).*

### 🏎️ Performance Summary

As expected with GPU acceleration, there is a modest startup cost for smaller matrices, but PfCUDA exhibits increasingly strong performance as matrix size grows:

* **Crossover Point:** PfCUDA overtakes Lrux at matrix sizes of roughly **1700 × 1700**.
* **Peak Speedup:** Delivers up to **~6.7× speedup** versus Lrux at **3000 × 3000**.
* **Average Speedup:** Averages **~1.9× speedup** over the measured size range.
* **Numerical Stability:** Both implementations show excellent numerical agreement, with accuracy errors consistently `≤ 1e-11`.

---

## 🛠️ Implementation Details

### CUDA + JAX FFI

The GPU path (`pfcuda/cuda_api.py`) uses JAX custom JVP definitions to wrap compiled CUDA kernels.
**Source files:** `src/pfaffian.cu`, `src/pfaffian_sm.cu`, `src/slog_pfaffian.cu`, `src/slog_pfaffian_lg.cu`, and `bindings/jax_bindings.cu`.

### C++ + pybind11

The native CPU implementation is exposed through a pybind11 module.
**Source files:** `bindings/pybind_bindings.cpp`, `src/pfaffian_cpu.cpp`.

### Pure Python

Available in `pfcuda/pfaffian_py.py` for portability, validation, and fallback usage without compilation requirements.

---

## ✅ Testing & Requirements

### Running Tests

Ensure your environment is working correctly by running the test suite:

```bash
pytest test/

```

### Requirements

* **Python:** `>= 3.9`
* **Core Libs:** `jax`, `jaxlib`, `numpy`
* **Build Tools:** `pybind11`, `CMake >= 3.18`
* **Hardware:** CUDA toolkit for GPU support

---

## 📂 Project Structure

* `pfcuda/` — Python package entry points and API wrappers.
* `include/` — CUDA/C++ header files.
* `src/` — Core CUDA and CPU source code.
* `bindings/` — JAX and pybind11 binding sources.
* `benchmarking/` — Scripts for comparative performance analysis.
* `test/` — Unit tests for Pfaffian mathematical behavior.

---

## 🤝 Contributing

Contributions are highly encouraged! Feel free to open an issue or submit a pull request if you want to improve GPU support, broaden data type coverage, or extend benchmark comparisons.

---

## 📝 Citation

If you use PfCUDA in your research or project, please cite the author and original bachelor thesis:

> **Muhammad Mouiz Ghouri** > *"Optimized Pfaffian Computation and Its Differentiation: From CPU Implementations to GPU Acceleration"* > Eötvös Loránd University, Budapest, Hungary, 2026.