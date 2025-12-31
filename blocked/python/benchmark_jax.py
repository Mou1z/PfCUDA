import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
# from pfaffian_jax import pfaffian  # Your JAX Pfaffian function
from pfaffian_jax_naive import pfaffian
# from pfaffian_det_based import pfaffian

# ----------------------------
# BENCHMARK CONFIG
# ----------------------------
RESULTS_FILE = "pfaffian_benchmark.json"

MATRIX_SIZES = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 5000]
RUNS_PER_SIZE = 5

PLOT_DIR = "benchmarks"
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------
# UTILITIES
# ----------------------------
def random_skew_symmetric(n):
    A = np.random.randn(n, n)
    return A - A.T

def time_jax_function(func, A, runs):
    # Convert numpy array to jax array
    A_jax = jnp.array(A, dtype=jnp.float64)
    # Warm-up to trigger JIT compilation
    func(A_jax).block_until_ready()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(A_jax)
        # Make sure computation finishes (important for JAX)
        result.block_until_ready()
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

print("Running JAX Pfaffian benchmark...")
for n in MATRIX_SIZES:
    print(f"  n = {n}")
    A = random_skew_symmetric(n)
    mean_t, std_t = time_jax_function(pfaffian, A, RUNS_PER_SIZE)

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
        linestyle="--",
        label=None
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
plt.title("JAX Pfaffian Runtime Benchmark")
plt.grid(True)
plt.legend(title="Run date")
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "pfaffian_benchmark.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")
