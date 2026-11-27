from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_iv_pv_curves(iv_curve: np.ndarray, pv_curve: np.ndarray, output_path: str | Path, title: str = "PV Module I-V and P-V Characteristics") -> str:
    """Plot I-V and P-V curves with characteristic points marked."""
    v = iv_curve[:, 0]
    i = iv_curve[:, 1]
    p = pv_curve[:, 0] if pv_curve.ndim == 2 else pv_curve

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Voltage (V)", fontsize=12)
    ax1.set_ylabel("Current (A)", color=color, fontsize=12)
    ax1.plot(v, i, color=color, linewidth=2.5, label="I-V Curve")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Isc and Voc markers
    ax1.scatter([0], [i[0]], color="blue", s=80, zorder=5, label=f"Isc = {i[0]:.2f} A")
    ax1.scatter([v[-1]], [0], color="navy", s=80, zorder=5, label=f"Voc = {v[-1]:.2f} V")

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Power (W)", color=color, fontsize=12)
    ax2.plot(v, p, color=color, linewidth=2.5, linestyle="--", label="P-V Curve")
    ax2.tick_params(axis="y", labelcolor=color)

    # MPP marker
    max_idx = np.argmax(p)
    ax2.scatter([v[max_idx]], [p[max_idx]], color="red", s=100, marker="*", zorder=6, label=f"Pmp = {p[max_idx]:.1f} W")

    plt.title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=150)
    plt.close(fig)
    return str(out_p)
