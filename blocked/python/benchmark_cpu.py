import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pfaffian_np_single_block import pfaffian


# ----------------------------
# BENCHMARK CONFIG
# ----------------------------
RESULTS_FILE = "pfaffian_benchmark.json"

MATRIX_SIZES = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]
RUNS_PER_SIZE = 5

PLOT_DIR = "benchmarks"


# ----------------------------
# UTILITIES
# ----------------------------
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


# ----------------------------
# RUN BENCHMARK
# ----------------------------
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
# PLOTTING
# ----------------------------
plt.figure(figsize=(10, 6))

for run in previous_results[:-1]:
    plt.plot(
        run["sizes"],
        run["mean_times"],
        linewidth=1,
        alpha=0.4,
        label=None,
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

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(PLOT_DIR, f"pfaffian_benchmark.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")