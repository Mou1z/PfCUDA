import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Check if cupy is available for synchronization
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import pfaffian_cuda

# -----------------------------
# Configuration
# -----------------------------
SIZES = [10, 20, 30]
RUNS_PER_SIZE = 5000

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def generate_matrix(n: int) -> np.ndarray:
    """Generate an antisymmetric, Fortran-contiguous matrix."""
    A = np.random.randn(n, n)
    A = A - A.T
    return np.asfortranarray(A, dtype=np.float64)

def sync_device():
    if HAS_CUPY:
        cp.cuda.runtime.deviceSynchronize()

def warmup(func, A, repeats=3):
    for _ in range(repeats):
        func(A)
        sync_device()

def benchmark(func, A, runs=5):
    times = []
    for _ in range(runs):
        sync_device()
        start = time.perf_counter()
        func(A)
        sync_device()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
    }

# -----------------------------
# Run benchmark
# -----------------------------
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_results = {
    "timestamp": timestamp,
    "sizes": {},
}

print(f"Starting Pfaffian CUDA benchmark at {timestamp}\n")

for n in SIZES:
    print(f"Matrix size n = {n}")
    A = generate_matrix(n)
    
    warmup(pfaffian_cuda.pfaffian, A)
    stats = benchmark(pfaffian_cuda.pfaffian, A, RUNS_PER_SIZE)
    current_results["sizes"][str(n)] = stats

    print(f"  mean: {stats['mean']:.6f}s | std: {stats['std']:.6f}s")

# Save current result to JSON
result_filename = f"run_{timestamp}.json"
result_path = os.path.join(RESULTS_DIR, result_filename)

with open(result_path, "w") as f:
    json.dump(current_results, f, indent=2)

print(f"\nSaved results to {result_path}")

# -----------------------------
# Linear Scale Plotting
# -----------------------------
plt.figure(figsize=(10, 7), facecolor='white')

# 1. Plot Historical Runs
all_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])
historical_files = [f for f in all_files if f != result_filename]

for file in historical_files:
    with open(os.path.join(RESULTS_DIR, file), "r") as f:
        data = json.load(f)
        # Sort by size to ensure lines don't zig-zag
        raw_sizes = sorted([(int(k), v["mean"]) for k, v in data["sizes"].items()])
        h_sizes, h_means = zip(*raw_sizes)
        plt.plot(h_sizes, h_means, alpha=0.4, linewidth=1, zorder=1)

# 2. Plot Current Run
current_data = sorted([(int(k), v["mean"]) for k, v in current_results["sizes"].items()])
c_sizes, c_means = zip(*current_data)
plt.plot(c_sizes, c_means, marker="o", markersize=3, 
         linewidth=1.7, label="Current Run", zorder=5)

# 3. Clean "Regular" Grid and Spacing
# We remove plt.xscale("log") to make it linear (the default)
plt.grid(True, linestyle="--", linewidth=0.5, color='#dcdde1', alpha=0.8)

# Start axes at 0 for an honest linear representation
plt.xlim(left=0)
plt.ylim(bottom=0)

# Aesthetics
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Matrix Size (n)", fontsize=11, fontweight='bold')
plt.ylabel("Execution Time (seconds)", fontsize=11, fontweight='bold')
plt.title("Pfaffian Benchmark: Linear Scaling", pad=20, fontsize=14)

plt.legend(frameon=False)
plt.tight_layout()

plot_path = os.path.join(PLOTS_DIR, "benchmark_linear.png")
plt.savefig(plot_path, dpi=300)
plt.show()