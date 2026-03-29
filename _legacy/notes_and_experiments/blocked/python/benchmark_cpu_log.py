import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
# from pfaffian_simple import pfaffian
# from pfaffian_np_single_block_op import pfaffian
# from pfaffian_np_single_block import pfaffian
from pfaffian_op import pfaffian

# ----------------------------
# BENCHMARK CONFIG
# ----------------------------
RESULTS_FILE = "pfaffian_benchmark.json"

MATRIX_SIZES = range(2, 1051, 50)
RUNS_PER_SIZE = 7

PLOT_DIR = "benchmarks"
os.makedirs(PLOT_DIR, exist_ok=True)


def random_skew_symmetric(n):
    A = np.random.randn(n, n)
    return A - A.T


def time_function(func, A, runs):
    times = []
    for _ in range(runs):
        A_copy = A.copy()
        start = time.perf_counter()
        func(A_copy)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


previous_results = load_results()

current_run = {
    "timestamp": datetime.now().isoformat(),
    "sizes": [],
    "mean_times": [],
    "std_times": []
}

print("Running Pfaffian benchmark...")
for n in MATRIX_SIZES:
    print(f"  n = {n}")
    A = random_skew_symmetric(n)
    mean_t, std_t = time_function(pfaffian, A, RUNS_PER_SIZE)

    current_run["sizes"].append(n)
    current_run["mean_times"].append(mean_t)
    current_run["std_times"].append(std_t)

previous_results.append(current_run)
save_results(previous_results)

# ----------------------------
# Plot standard runtime
# ----------------------------
plt.figure(figsize=(10, 6))
for run in previous_results[:-1]:
    plt.plot(
        run["sizes"],
        run["mean_times"],
        linewidth=1,
        alpha=0.4,
        linestyle="--",
    )

run = previous_results[-1]
plt.plot(
    run["sizes"],
    run["mean_times"],
    marker="o",
    linewidth=3,
    label=f"Current run ({run['timestamp'].split('T')[0]})"
)

plt.xlabel("Matrix size (n)")
plt.ylabel("Execution time (seconds)")
plt.title("Pfaffian Runtime Benchmark")
plt.grid(True)
plt.legend(title="Run date")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"pfaffian_benchmark.png"), dpi=300)

# ----------------------------
# Log-Log plot and constant C estimation
# ----------------------------
log_sizes = np.log(np.array(run["sizes"]))
log_times = np.log(np.array(run["mean_times"]))

# Linear fit in log-log space
slope, intercept = np.polyfit(log_sizes, log_times, 1)
C_est = np.exp(intercept)

print(f"\n--- Log-Log Fit ---")
print(f"Slope (expected ~3): {slope:.3f}")
print(f"Intercept: {intercept:.5f}")
print(f"Estimated constant factor C: {C_est:.5e}")

# Plot log-log
plt.figure(figsize=(10, 6))
plt.plot(log_sizes, log_times, 'o', label='Data')
plt.plot(log_sizes, slope * log_sizes + intercept, '-', label=f'Fit: slope={slope:.2f}')
plt.xlabel("log(Matrix size n)")
plt.ylabel("log(Execution time)")
plt.title("Pfaffian Log-Log Benchmark")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"pfaffian_benchmark_loglog.png"), dpi=300)
print(f"Log-log plot saved to {os.path.join(PLOT_DIR, f'pfaffian_benchmark_loglog.png')}")