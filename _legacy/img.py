import numpy as np
import matplotlib.pyplot as plt

def generate_skew_symmetric(n):
    if n % 2 != 0:
        raise ValueError("n must be even")

    A = np.zeros((n, n), dtype=int)
    counter = 1

    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = counter
            A[j, i] = -counter
            counter += 1

    return A


def plot_matrix_with_numbers(A, save_path=None):
    n = A.shape[0]

    # ---- Dynamic Scaling ----
    base_cell_size = 0.5          # controls overall image growth
    fig_size = max(6, n * base_cell_size)

    font_size = max(4, 180 / n)   # shrink font as matrix grows
    tick_size = max(4, 150 / n)

    plt.figure(figsize=(fig_size, fig_size), dpi=200)
    plt.imshow(A)

    # Overlay numbers
    for i in range(n):
        for j in range(n):
            plt.text(
                j, i, str(A[i, j]),
                ha='center', va='center',
                fontsize=font_size
            )

    plt.xticks(range(n), fontsize=tick_size)
    plt.yticks(range(n), fontsize=tick_size)

    plt.title(f"{n}x{n} Skew-Symmetric Matrix", fontsize=max(10, 300/n))
    plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    n = 34  # try 34 or larger
    A = generate_skew_symmetric(n)
    plot_matrix_with_numbers(A, save_path="skew_matrix_scaled.png")
