from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_thermal_image(
    thermal_img: np.ndarray,
    output_path: str | Path,
    colormap: str = "inferno",
    title: str = "Thermal Infrared Image",
) -> str:
    """Plot thermal image with temperature scale bar."""
    fig, ax = plt.subplots(figsize=(7, 6))

    img = thermal_img.squeeze()
    im = ax.imshow(img, cmap=colormap)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Thermal Intensity", fontsize=11)

    fig.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=150)
    plt.close(fig)
    return str(out_p)
