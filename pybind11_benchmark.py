import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Import your modules here
# from your_module import slog_pfaffian as my_pf
import lrux
from pfcuda import slog_pfaffian_f64 as my_pf


def benchmark_fn(fn, matrix, n_runs=10):
    
    if fn == lrux.slogpf:
        a, b = fn(matrix.copy())

        a.block_until_ready()
        b.block_until_ready()

        start = time.perf_counter()

        for _ in range(n_runs):
            a, b = fn(matrix.copy())

            a.block_until_ready()
            b.block_until_ready()

        end = time.perf_counter()
    else:
        res = fn(matrix.copy())

        start = time.perf_counter()

        for _ in range(n_runs):
            a, b = fn(matrix.copy())

        end = time.perf_counter()
    return (end - start) / n_runs


def generate_skew_symmetric(n):
    mat = np.random.randn(n, n)
    return mat - mat.T


def run_benchmarks(sizes):
    my_times = []
    lrux_times = []

    print(f"{'N':>6} | {'My Impl (s)':>12} | {'lrux (s)':>12}")
    print("-" * 35)

    for n in sizes:
        if n % 2 != 0:
            print(f"{n:6d} | Skipping (Pfaffian requires even size)")
            continue

        matrix = generate_skew_symmetric(n)

        try:
            t_my = benchmark_fn(my_pf, matrix.copy())
            t_lrux = benchmark_fn(lrux.slogpf, matrix.copy())

            lrux_times.append(t_lrux)
            my_times.append(t_my)

            print(f"{n:6d} | {t_my:12.6f} | {t_lrux:12.6f}")

        except Exception as e:
            print(f"{n:6d} | Error at this size: {e}")
            break

    return my_times, lrux_times


def plot_results(sizes, my_times, lrux_times):
    os.makedirs("benchmarks", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes[:len(my_times)], my_times, 'o-', label='My Impl (pybind11)')
    plt.plot(sizes[:len(lrux_times)], lrux_times, 's-', label='lrux')

    plt.title("Pfaffian Benchmark: Custom vs lrux")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale('log')
    plt.grid(True, ls="-", alpha=0.5)
    plt.legend()

    save_path = "benchmarks/pfaffian_benchmark.png"
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")


if __name__ == "__main__":
    matrix_sizes = range(100, 1501, 100)

    m_t, l_t = run_benchmarks(matrix_sizes)
    plot_results(matrix_sizes, m_t, l_t)