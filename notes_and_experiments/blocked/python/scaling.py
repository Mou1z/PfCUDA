import os
import json
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "benchmarks"

# Load your data
# Assuming the data is saved as 'benchmarks.json'
with open('pfaffian_benchmark.json', 'r') as f:
    data = json.load(f)

sizes = np.array(data[0]['sizes'])
t1 = np.array(data[0]['mean_times'])
t2 = np.array(data[1]['mean_times'])

# Define a theoretical n^3 scaling line
# We normalize it to start at the same point as Imp 1 for visual comparison
# T_ref = C * n^3 => C = T_initial / (n_initial^3)
c_ref = t1[0] / (sizes[0]**3)
t_n3 = c_ref * (sizes**3)

plt.figure(figsize=(10, 6))

# Plotting on Log-Log scale
plt.loglog(sizes, t1, 'o-', label='Implementation 1', linewidth=2)
plt.loglog(sizes, t2, 's-', label='Implementation 2', linewidth=2)
plt.loglog(sizes, t_n3, '--', label='Theoretical $n^3$ Scaling', color='gray')

# Formatting
plt.title('Pfaffian Scaling vs $n^3$ Complexity', fontsize=14)
plt.xlabel('Matrix Size (n) [Log Scale]', fontsize=12)
plt.ylabel('Mean Time (s) [Log Scale]', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Estimate the constants C where T = C * n^3
# We use the last (largest) data point for the most accurate asymptotic constant
c1 = t1[-1] / (sizes[-1]**3)
c2 = t2[-1] / (sizes[-1]**3)

print(f"Estimated Constants (T = C * n^3):")
print(f"Implementation 1: C ≈ {c1:.2e}")
print(f"Implementation 2: C ≈ {c2:.2e}")
print(f"Imp 2 constant is {c1/c2:.2f}x smaller than Imp 1 constant at n={sizes[-1]}")

plot_path = os.path.join(PLOT_DIR, f"pfaffian_scaling.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")