from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_roc_pr_curves(y_true: np.ndarray, y_pred_probs: np.ndarray, class_names: list[str], output_dir: str | Path) -> tuple[str, str]:
    """Plot ROC and Precision-Recall curves for all fault classes."""
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    roc_path = out_d / "roc_curves.png"
    pr_path = out_d / "pr_curves.png"

    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 1)))

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        if i < y_true.shape[1] and i < y_pred_probs.shape[1]:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Multiclass ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(str(roc_path), dpi=150)
    plt.close(fig)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        if i < y_true.shape[1] and i < y_pred_probs.shape[1]:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=colors[i], lw=2, label=f"{class_names[i]} (AP = {pr_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(str(pr_path), dpi=150)
    plt.close(fig)

    return str(roc_path), str(pr_path)
