import os
import jax
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from lrux import slogpf
from pfcuda import pfaffian as pfaffian_f64

DATA_FILE = "benchmarks/pfaffian_data.json"

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', "cuda")

def generate_skew_symmetric(n, seed=None):
    """Generate a random skew-symmetric matrix."""
    if seed is not None:
        np.random.seed(seed)
    mat = np.random.randn(n, n).astype(np.float64)
    return mat - mat.T


def benchmark_pfcuda(matrix, n_runs=1000, warmup=1):
    """Benchmark the pfcuda pfaffian function."""
    for _ in range(warmup):
        pfaffian_f64(matrix.copy())

    times = []
    results = []

    for _ in range(n_runs):
        mat_copy = matrix.copy()

        start = time.perf_counter()
        value = pfaffian_f64(mat_copy)
        jax.block_until_ready(value)
        end = time.perf_counter()

        times.append(end - start)
        results.append(float(value))

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, results, times


def benchmark_lrux(matrix, n_runs=1000, warmup=1):
    """Benchmark the lrux slogpf function."""
    for _ in range(warmup):
        slogpf(matrix.copy())

    times = []
    results = []

    for _ in range(n_runs):
        mat_copy = matrix.copy()

        start = time.perf_counter()
        sign, log_abs = slogpf(mat_copy)
        jax.block_until_ready(sign)
        jax.block_until_ready(log_abs)
        end = time.perf_counter()

        times.append(end - start)
        results.append((int(sign), float(log_abs)))

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, results, times


def run_benchmarks(sizes, benchmark_func, function_name, n_runs=1000):
    """Run benchmarks for various matrix sizes using a specific benchmark function."""
    results_data = {
        'sizes': [],
        'avg_times': [],
        'std_times': [],
        'values': [],
        'signs': [],
        'log_abs_values': [],
        'expected_abs_pf': [],
        'accuracy_errors': []
    }

    print(f"{'N':>6} | {'Avg Time (ms)':>15} | {'Std (ms)':>12} | {'Value/LogAbs':>22} | {'Abs Error':>15}")
    print("-" * 104)

    for n in sizes:
        if n % 2 != 0:
            print(f"{n:6d} | Skipping (Pfaffian requires even size)")
            continue

        try:
            matrix = generate_skew_symmetric(n)
            _, logdet = np.linalg.slogdet(matrix)
            expected_log_abs = 0.5 * logdet

            avg_time, std_time, runs_results, _ = benchmark_func(matrix, n_runs=n_runs)

            if function_name == 'Lrux':
                sign, log_abs = runs_results[0]
                value = sign * np.exp(log_abs)
                accuracy_error = log_abs - expected_log_abs
                value_display = f"{sign:2d},{log_abs:12.4e}"
                results_data['log_abs_values'].append(log_abs)
            else:
                value = float(runs_results[0])
                log_abs = np.log(abs(value)) if value != 0 else -np.inf
                accuracy_error = log_abs - expected_log_abs
                value_display = f"{value:20.4e}"
                results_data['log_abs_values'].append(log_abs)

            results_data['sizes'].append(n)
            results_data['avg_times'].append(avg_time * 1000)
            results_data['std_times'].append(std_time * 1000)
            results_data['values'].append(value)
            results_data['signs'].append(np.sign(value))
            results_data['expected_abs_pf'].append(np.exp(expected_log_abs))
            results_data['accuracy_errors'].append(accuracy_error)

            print(f"{n:6d} | {avg_time*1000:15.4f} | {std_time*1000:12.4f} | {value_display:>22} | {accuracy_error:15.4e}")

        except Exception as e:
            print(f"{n:6d} | Error: {str(e)[:50]}")
            break

    return results_data


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
        'values': results_data['values'],
        'expected_abs_pf': results_data['expected_abs_pf'],
        'accuracy_errors': results_data['accuracy_errors']
    }

    data['runs'].append(run)
    return data


def plot_all_benchmarks(data):
    """Plot all benchmarks from data file."""
    os.makedirs("benchmarks", exist_ok=True)

    colors = {'PfCUDA': 'blue', 'Lrux': 'orange'}
    markers = {'PfCUDA': 'o', 'Lrux': 's'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    legend_added = set()

    for run in data['runs']:
        function = run['function']
        color = colors.get(function, 'gray')
        marker = markers.get(function, 'x')

        sizes = run['sizes']
        avg_times = run['avg_times']
        std_times = run['std_times']
        accuracy_errors = run.get('accuracy_errors', [])

        label = f"{function}" if function not in legend_added else ""
        ax1.errorbar(sizes, avg_times, yerr=std_times, fmt=f'{marker}-',
                     capsize=3, label=label, color=color, alpha=0.7, markersize=6)

        if accuracy_errors:
            ax2.plot(sizes, accuracy_errors, f'{marker}-', label=label,
                     color=color, alpha=0.7, markersize=6)

        legend_added.add(function)

    ax1.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Pfaffian Benchmark: Execution Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Log-Accuracy Error (log|Pf| - expected)', fontsize=12, fontweight='bold')
    ax2.set_title('Pfaffian Log-Accuracy Error', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    save_path = "benchmarks/pfaffian_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Comparison graph saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    matrix_sizes = range(2, 33, 2)

    print("Benchmarking PfCUDA pfaffian_f64")
    print("="*110)
    pfcuda_results = run_benchmarks(matrix_sizes, benchmark_pfcuda, 'PfCUDA', n_runs=1000)
    print_summary(pfcuda_results)

    print("\nBenchmarking Lrux slogpf")
    print("="*110)
    lrux_results = run_benchmarks(matrix_sizes, benchmark_lrux, 'Lrux', n_runs=1000)
    print_summary(lrux_results)

    all_data = load_or_create_data()
    all_data = add_run(all_data, pfcuda_results, 'PfCUDA')
    all_data = add_run(all_data, lrux_results, 'Lrux')
    save_data(all_data)

    print(f"\n✓ Data saved to {DATA_FILE}")
    plot_all_benchmarks(all_data)
