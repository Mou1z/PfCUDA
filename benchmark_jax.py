import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import your modules here
# from your_module import slog_pfaffian as my_pf
import lrux
from pfaffian import slog_pfaffian as my_pf

def benchmark_fn(fn, matrix, n_runs=10):    
    start = time.perf_counter()
    for _ in range(n_runs):
        res = fn(matrix.copy())
        res[0].block_until_ready()
        res[1].block_until_ready()

    end = time.perf_counter()
    
    return (end - start) / n_runs

def run_benchmarks(sizes):
    key = jax.random.PRNGKey(0)
    my_times = []
    lrux_times = []

    print(f"{'N':>6} | {'My CUDA (s)':>12} | {'lrux (s)':>12}")
    print("-" * 35)

    for n in sizes:
        key, subkey = jax.random.split(key)
        # Matrix must be skew-symmetric
        mat = jax.random.normal(subkey, (n, n))
        matrix = mat - mat.T 
        
        try:
            t_lrux = benchmark_fn(lrux.slogpf, matrix)
            t_my = benchmark_fn(my_pf, matrix)
            
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
    plt.plot(sizes, my_times, 'o-', label='My CUDA (FFI)')
    plt.plot(sizes, lrux_times, 's-', label='lrux')
    
    plt.title("Pfaffian Benchmark: Custom FFI vs lrux")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale('log') # Log scale is often better for complexity scaling
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    save_path = "benchmarks/pfaffian_benchmark.png"
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")

if __name__ == "__main__":
    # Define range of matrix sizes (must be even for Pfaffian)
    matrix_sizes = range(100, 2001, 100)
    
    try:
        m_t, l_t = run_benchmarks(matrix_sizes)
        plot_results(matrix_sizes, m_t, l_t)
    except NameError as e:
        print(f"Error: {e}. Please ensure 'my_slog_pfaffian' and 'lrux' are imported.")