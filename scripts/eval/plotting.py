from __future__ import annotations

from pathlib import Path


def save_confusion_matrix_png(
    matrix: list[list[int]],
    labels: list[str],
    out_path: Path,
    title: str = "Matrice de confusion",
) -> None:
    """Sauve la matrice de confusion en PNG (matplotlib).

    Garde la dependance matplotlib hors du module metrics pour qu'il reste
    importable depuis l'environnement de test sans matplotlib installe.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(matrix)
    fig, ax = plt.subplots(
        figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.4))
    )
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predit")
    ax.set_ylabel("Verite")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_latency_distribution_png(
    latencies_ms: list[float],
    out_path: Path,
    title: str = "Distribution latence LLM",
) -> None:
    """Histogramme de la distribution latence LLM."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies_ms, bins=30, color="#3b82f6", edgecolor="white")
    ax.set_xlabel("Latence (ms)")
    ax.set_ylabel("Nombre d'appels")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_latency_comparison_png(
    latencies_by_backend: dict[str, list[float]],
    out_path: Path,
    title: str = "Distribution latence par backend LLM",
) -> None:
    """Boxplot side-by-side : un backend par boite (slice 5 #75).

    Met en evidence la difference d'ordre de grandeur Gemma3:4b CPU (~10^5 ms)
    vs Mistral Small managed (~10^3 ms) sur les memes inputs. L'echelle est
    log si le ratio max/min depasse 10 pour rester lisible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(latencies_by_backend.keys())
    data = [latencies_by_backend[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    # matplotlib >=3.9 : `labels` renomme en `tick_labels` (drop en 3.11).
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("Latence (ms)")
    ax.set_title(title)

    flat = [v for series in data for v in series if v > 0]
    if flat and (max(flat) / max(min(flat), 1e-9)) > 10:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
