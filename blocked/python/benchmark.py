import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
MATRIX_SIZES = [50, 100, 150, 200, 250, 300, 500, 1000]
REPEATS = 5
JSON_FILE = "benchmark_results.json"


def generate_skew_symmetric(n, backend="numpy", key=None):
    """
    Returns a random n x n skew-symmetric matrix.
    backend: 'numpy' or 'jax'
    """
    if backend == "jax":
        import jax
        import jax.numpy as jnp
        assert key is not None, "JAX backend requires PRNGKey"
        A = jax.random.normal(key, (n, n))
        return A - A.T
    else:
        A = np.random.randn(n, n)
        return A - A.T


def benchmark_function(func, n, backend="numpy", key=None):
    """
    Benchmarks a single function on a single matrix size.
    Handles warm-up and multiple repeats.
    """
    A = generate_skew_symmetric(n, backend=backend, key=key)

    # Convert to NumPy array if the function is NumPy-only
    if backend == "jax" and "numpy" in func.__module__:
        A = np.array(A)

    # Warm-up
    func(A)

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        func(A)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = float(np.mean(times))
    return avg_time


def run_benchmarks(func, backend="numpy"):
    """
    Benchmarks a function over all MATRIX_SIZES.
    backend = 'numpy' or 'jax'
    """
    results = {}
    for n in MATRIX_SIZES:
        print(f"Benchmarking n={n}...")
        key = None
        if backend == "jax":
            import jax
            key = jax.random.PRNGKey(0)
        t = benchmark_function(func, n, backend=backend, key=key)
        results[n] = t
        print(f"  -> {t:.6f} seconds")
    return results


def save_results(new_results):
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    run_id = str(int(time.time()))
    data[run_id] = {str(k): v for k, v in new_results.items()}

    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {JSON_FILE}")


def plot_results(latest_results):
    plt.figure(figsize=(10, 6))

    xs = list(map(int, latest_results.keys()))
    ys = [latest_results[str(k)] for k in xs]

    xs, ys = zip(*sorted(zip(xs, ys)))
    plt.plot(xs, ys, marker="o", linewidth=3, label="Current Run")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
        for run_id, r in data.items():
            xs_prev = list(map(int, r.keys()))
            ys_prev = [r[str(k)] for k in xs_prev]
            xs_prev, ys_prev = zip(*sorted(zip(xs_prev, ys_prev)))
            plt.plot(xs_prev, ys_prev, linestyle="--", alpha=0.4, label=f"Run {run_id}")

    plt.xlabel("Matrix size (n)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Pfaffian Benchmark")
    plt.grid(True)
    plt.legend()
    plt.savefig("benchmark_plot.png", dpi=200)
    print("Graph saved to benchmark_plot.png")
    plt.show()


if __name__ == "__main__":
    # from pfaffian_np_single_block import pfaffian
    # from pfapack.pfaffian import pfaffian
    from pfaffian_jax import pfaffian

    backend = "numpy"

    results = run_benchmarks(pfaffian, backend=backend)
    results_str_keys = {str(k): v for k, v in results.items()}
    save_results(results_str_keys)
    plot_results(results_str_keys)