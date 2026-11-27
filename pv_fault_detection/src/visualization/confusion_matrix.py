from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: str | Path, title: str = "Multilabel Confusion Matrix") -> str:
    """Plot and save normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Normalize if not already normalized
    if cm.sum() > 0 and cm.max() > 1.0:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    else:
        cm_norm = cm

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=150)
    plt.close(fig)
    return str(out_p)
