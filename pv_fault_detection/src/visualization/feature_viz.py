from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_feature_embeddings(
    z_features: np.ndarray,
    labels: list[str] | np.ndarray,
    domain_ids: list[str] | np.ndarray | None = None,
    output_path: str | Path = "reports/tsne_embeddings.png",
    use_tsne: bool = True,
) -> str:
    """Plot 2D embedding of bottleneck features z colored by class and domain."""
    n_samples = len(z_features)
    if n_samples < 4:
        z_2d = np.random.randn(n_samples, 2)
    elif use_tsne and n_samples >= 10:
        perplexity = min(30.0, max(5.0, float(n_samples - 1) / 3.0))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        z_2d = tsne.fit_transform(z_features)
    else:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Colored by Fault Label
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    for idx, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax1.scatter(z_2d[mask, 0], z_2d[mask, 1], label=str(lbl), color=colors[idx], alpha=0.7, edgecolors="none")

    ax1.set_title("Bottleneck z: Clustered by Fault Class", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9, framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Colored by Domain
    if domain_ids is not None:
        unique_domains = sorted(list(set(domain_ids)))
        d_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_domains), 1)))
        for idx, dom in enumerate(unique_domains):
            mask = np.array([d == dom for d in domain_ids])
            ax2.scatter(z_2d[mask, 0], z_2d[mask, 1], label=str(dom), color=d_colors[idx], alpha=0.7, edgecolors="none")
        ax2.set_title("Bottleneck z: Clustered by Source Domain", fontsize=12, fontweight="bold")
        ax2.legend(loc="best", fontsize=9, framealpha=0.8)
    else:
        ax2.set_title("Bottleneck z: Density Map", fontsize=12, fontweight="bold")
        ax2.hexbin(z_2d[:, 0], z_2d[:, 1], gridsize=20, cmap="viridis")

    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=150)
    plt.close(fig)
    return str(out_p)
