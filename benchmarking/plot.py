"""Two-panel comparison figure: latency on the left, accuracy on the right."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "PfCUDA": ("tab:blue", "o"),
    "lrux": ("tab:orange", "s"),
    "TorchPfaffian": ("tab:green", "^"),
    "PyPfaffian": ("tab:red", "v"),
    "pfapack": ("tab:gray", "d"),
}

# Errors at or below this are indistinguishable from the reference's own
# roundoff; floored so they remain visible on a log axis.
ERROR_FLOOR = 1e-16


def _series(results):
    grouped = {}
    for r in results:
        if r["latency_ms"] is None:
            continue
        grouped.setdefault(r["implementation"], []).append(r)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["n"])
    return grouped


def render(payload, path):
    grouped = _series(payload["results"])
    env = payload["environment"]

    fig, (ax_time, ax_err) = plt.subplots(1, 2, figsize=(15, 5.5))

    for name, rows in grouped.items():
        colour, marker = STYLE.get(name, ("black", "x"))
        sizes = [r["n"] for r in rows]
        ax_time.plot(sizes, [r["latency_ms"] for r in rows], marker=marker,
                     color=colour, label=name, markersize=5, alpha=0.85)
        ax_err.plot(sizes, [max(r["mean_log_error"], ERROR_FLOOR) for r in rows],
                    marker=marker, color=colour, label=name, markersize=5, alpha=0.85)

    ax_time.set_xlabel("Matrix size (N x N)")
    ax_time.set_ylabel("Median latency per call (ms)")
    ax_time.set_title(f"{payload['suite']}: time per call, lower is better")
    ax_time.set_yscale("log")

    ax_err.set_xlabel("Matrix size (N x N)")
    ax_err.set_ylabel(r"Mean $|\log|Pf|$ error$|$ vs. slogdet")
    ax_err.set_title("Agreement with 0.5 x logabsdet, lower is better")
    ax_err.set_yscale("log")
    ax_err.axhline(ERROR_FLOOR, color="black", linestyle=":", linewidth=1,
                   label="double precision floor")

    for ax in (ax_time, ax_err):
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"{payload['samples']} distinct matrices per size, float64, "
        f"{env['gpu']} | pfcuda {env['versions'].get('pfcuda', '?')} "
        f"@ {env['git_rev']}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
