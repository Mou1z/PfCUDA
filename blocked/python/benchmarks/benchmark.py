import time
import json
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pfaffian_jax import pfaffian

MATRIX_SIZES = [50, 100, 150, 200, 250, 300, 500, 1000, 1500]
REPEATS = 5

JSON_FILE = "benchmark_results.json"


def benchmark_once(n):
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (n, n))
    A = A - A.T

    pfaffian(A).block_until_ready()

    times = []
    for _ in range(REPEATS):
        start = time.time()
        result = pfaffian(A)
        result.block_until_ready()
        end = time.time()
        times.append(end - start)

    return float(sum(times) / len(times))


def run_benchmarks():
    results = {}
    for n in MATRIX_SIZES:
        print(f"Benchmarking n={n}...")
        t = benchmark_once(n)
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
    data[run_id] = new_results

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

        run_ids = list(data.keys())
        if len(run_ids) > 1:
            for run_id in run_ids[:-1]:
                r = data[run_id]
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
    results = run_benchmarks()
    results_str_keys = {str(k): v for k, v in results.items()}  # JSON requires string keys

    save_results(results_str_keys)
    plot_results(results_str_keys)
