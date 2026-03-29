import os
import json
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "benchmarks"
# --- ADJUST THIS VARIABLE ---
POWER = 3  # Set to 1 for linear, 2 for quadratic, 3 for cubic, etc.
# ----------------------------

# Load your data
with open('pfaffian_benchmark.json', 'r') as f:
    data = json.load(f)

sizes = np.array(data[0]['sizes'])
t1 = np.array(data[0]['mean_times'])
t2 = np.array(data[1]['mean_times'])

# Define a theoretical scaling line based on the POWER variable
# T_ref = C * n^POWER
DIVISOR = 3  # e.g. 3 for n^3 / 3

c_ref = t1[0] / ((sizes[0]**POWER) / DIVISOR)
t_ref = c_ref * ((sizes**POWER) / DIVISOR)

plt.figure(figsize=(10, 6))

# Plotting on Log-Log scale
plt.loglog(sizes, t1, 'o-', label='Implementation 1', linewidth=2)
plt.loglog(sizes, t2, 's-', label='Implementation 2', linewidth=2)
plt.loglog(sizes, t_ref, '--', label=f'Theoretical $n^{POWER}$ Scaling', color='gray')

# Formatting
plt.title(f'Pfaffian Scaling vs $n^{POWER}$ Complexity', fontsize=14)
plt.xlabel('Matrix Size (n) [Log Scale]', fontsize=12)
plt.ylabel('Mean Time (s) [Log Scale]', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Estimate the constants C where T = C * n^POWER
c1 = t1[-1] / (sizes[-1]**POWER)
c2 = t2[-1] / (sizes[-1]**POWER)

print(f"Comparison against O(n^{POWER}):")
print(f"Implementation 1 constant: {c1:.2e}")
print(f"Implementation 2 constant: {c2:.2e}")

plot_path = os.path.join(PLOT_DIR, f"pfaffian_scaling_n{POWER}.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")