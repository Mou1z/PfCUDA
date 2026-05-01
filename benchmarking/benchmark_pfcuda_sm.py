import os
import jax
import math
import time
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from pfcuda import pfaffian

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', "cuda")

DATA_FILE = "benchmarks/pfaffian_data.json"

def generate_skew_symmetric(n, seed=None):
    """Generate a random skew-symmetric matrix."""
    if seed is not None:
        np.random.seed(seed)
    mat = np.random.randn(n, n).astype(np.float64)
    return mat - mat.T


def benchmark_pfcuda(matrix, n_runs=5, warmup=1):
    """Benchmark the pfcuda slog_pfaffian function."""
    # Warmup run
    for _ in range(warmup):
        pfaffian(matrix.copy())
    
    # Timed runs
    times = []
    results = []
    
    for _ in range(n_runs):
        mat_copy = matrix.copy()
        
        start = time.perf_counter()
        result = pfaffian(mat_copy)
        end = time.perf_counter()
        
        times.append(end - start)

        phase = math.copysign(1.0, result)
        log_abs = math.log(abs(result))
        
        results.append((float(phase), float(log_abs)))
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, results, times


def check_consistency(results):
    """Check if multiple runs give consistent results."""
    if len(results) < 2:
        return True, 0
    
    phases = [r[0] for r in results]
    log_abs_vals = [r[1] for r in results]
    
    # Check phase consistency (should be ±1)
    phase_diffs = np.abs(np.diff(phases))
    
    # Check log_abs consistency
    log_abs_std = np.std(log_abs_vals)
    
    return log_abs_std < 0.1, log_abs_std


def run_benchmarks(sizes, n_runs=5):
    """Run benchmarks for various matrix sizes."""
    results_data = {
        'sizes': [],
        'avg_times': [],
        'std_times': [],
        'log_abs_values': [],
        'phases': [],
        'consistency_std': [],
        'expected_log_abs': [],
        'accuracy_errors': []
    }
    
    print(f"{'N':>6} | {'Avg Time (ms)':>15} | {'Std (ms)':>12} | {'Phase':>8} | {'Log|Pf|':>12} | {'Accuracy Error':>15} | {'Consistency':>12}")
    print("-" * 110)
    
    for n in sizes:
        if n % 2 != 0:
            print(f"{n:6d} | Skipping (Pfaffian requires even size)")
            continue
        
        try:
            matrix = generate_skew_symmetric(n)
            sign, logdet = np.linalg.slogdet(matrix)
            expected_log_abs = 0.5 * logdet
            
            avg_time, std_time, runs_results, times = benchmark_pfcuda(matrix, n_runs=n_runs)
            
            phase, log_abs = runs_results[0]
            accuracy_error = log_abs - expected_log_abs
            consistent, consistency_std = check_consistency(runs_results)
            
            results_data['sizes'].append(n)
            results_data['avg_times'].append(avg_time * 1000)  # Convert to ms
            results_data['std_times'].append(std_time * 1000)
            results_data['log_abs_values'].append(log_abs)
            results_data['phases'].append(phase)
            results_data['consistency_std'].append(consistency_std)
            results_data['expected_log_abs'].append(expected_log_abs)
            results_data['accuracy_errors'].append(accuracy_error)
            
            consistency_str = "✓ Good" if consistency_std < 0.1 else "✗ Bad"
            
            print(f"{n:6d} | {avg_time*1000:15.4f} | {std_time*1000:12.4f} | {phase:8.1f} | {log_abs:12.4f} | {accuracy_error:15.4f} | {consistency_str:>12}")
        
        except Exception as e:
            print(f"{n:6d} | Error: {str(e)[:50]}")
            break
    
    return results_data


def plot_benchmark_results(results_data):
    """Plot benchmark results (deprecated - kept for compatibility)."""
    pass


def print_summary(results_data):
    """Print summary statistics."""
    sizes = results_data['sizes']
    avg_times = results_data['avg_times']
    
    if len(sizes) > 0:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Smallest matrix size: {sizes[0]}x{sizes[0]} - {avg_times[0]:.4f} ms")
        print(f"Largest matrix size:  {sizes[-1]}x{sizes[-1]} - {avg_times[-1]:.4f} ms")
        print(f"Average time (all):   {np.mean(avg_times):.4f} ms")
        print(f"Max time:             {np.max(avg_times):.4f} ms")
        print(f"Min time:             {np.min(avg_times):.4f} ms")
        print("="*50)


def load_or_create_data():
    """Load existing benchmark data or create new file."""
    os.makedirs("benchmarks", exist_ok=True)
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {'runs': []}


def save_data(data):
    """Save benchmark data to JSON file."""
    os.makedirs("benchmarks", exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def add_run(data, results_data, function_name):
    """Add a new benchmark run to the data."""
    timestamp = datetime.now().isoformat()
    
    run = {
        'timestamp': timestamp,
        'function': function_name,
        'sizes': results_data['sizes'],
        'avg_times': results_data['avg_times'],
        'std_times': results_data['std_times'],
        'log_abs_values': results_data['log_abs_values'],
        'phases': results_data['phases'],
        'consistency_std': results_data['consistency_std'],
        'expected_log_abs': results_data['expected_log_abs'],
        'accuracy_errors': results_data['accuracy_errors']
    }
    
    data['runs'].append(run)
    return data


def plot_all_benchmarks(data):
    os.makedirs("benchmarks", exist_ok=True)

    colors = {'pfcuda': 'blue', 'lrux': 'orange'}
    markers = {'pfcuda': 'o', 'lrux': 's'}

    # ---------- 1. Execution Time ----------
    plt.figure(figsize=(8, 6))
    legend_added = set()

    for run in data['runs']:
        function = run['function']
        color = colors.get(function, 'gray')
        marker = markers.get(function, 'x')

        label = function if function not in legend_added else ""
        plt.errorbar(
            run['sizes'],
            run['avg_times'],
            yerr=run['std_times'],
            fmt=f'{marker}-',
            capsize=3,
            label=label,
            color=color,
            alpha=0.7
        )
        legend_added.add(function)

    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()

    path1 = "benchmarks/execution_time.png"
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"✓ Saved {path1}")


    # ---------- 2. Log|Pf| ----------
    plt.figure(figsize=(8, 6))
    legend_added.clear()

    for run in data['runs']:
        function = run['function']
        color = colors.get(function, 'gray')
        marker = markers.get(function, 'x')

        label = function if function not in legend_added else ""
        plt.plot(
            run['sizes'],
            run['log_abs_values'],
            f'{marker}-',
            label=label,
            color=color,
            alpha=0.7
        )
        legend_added.add(function)

    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('log|Pf|')
    plt.title('Log Absolute Pfaffian')
    plt.grid(True, alpha=0.3)
    plt.legend()

    path2 = "benchmarks/log_pfaffian.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"✓ Saved {path2}")


    # ---------- 3. Accuracy Error ----------
    plt.figure(figsize=(8, 6))
    legend_added.clear()

    for run in data['runs']:
        if not run.get('accuracy_errors'):
            continue

        function = run['function']
        color = colors.get(function, 'gray')
        marker = markers.get(function, 'x')

        label = function if function not in legend_added else ""
        plt.plot(
            run['sizes'],
            run['accuracy_errors'],
            f'{marker}-',
            label=label,
            color=color,
            alpha=0.7
        )
        legend_added.add(function)

    plt.axhline(y=0, linestyle='--')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Accuracy Error')
    plt.title('Accuracy Error')
    plt.grid(True, alpha=0.3)
    plt.legend()

    path3 = "benchmarks/accuracy_error.png"
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"✓ Saved {path3}")


if __name__ == "__main__":
    # Benchmark sizes
    matrix_sizes = range(2, 33, 2)
    
    print("Benchmarking pfcuda slog_pfaffian_f64")
    print("="*110)
    
    # Run benchmarks
    results = run_benchmarks(matrix_sizes, n_runs=1000)
    print_summary(results)
    
    # Load existing data and add new run
    all_data = load_or_create_data()
    all_data = add_run(all_data, results, 'pfcuda')
    save_data(all_data)
    
    print(f"\n✓ Data saved to {DATA_FILE}")
    
    # Generate comparison graph with all runs
    plot_all_benchmarks(all_data)