from __future__ import annotations

from pathlib import Path


def test_save_latency_comparison_png_writes_a_real_png(tmp_path: Path) -> None:
    """Slice 5 (#75) : boxplot multi-backend (Gemma vs Mistral)."""
    from scripts.eval.plotting import save_latency_comparison_png

    out = tmp_path / "lat.png"
    save_latency_comparison_png(
        {
            "gemma3:4b (local)": [120_000.0, 180_000.0, 250_000.0, 210_000.0],
            "mistral-small (managed)": [2500.0, 3100.0, 4200.0, 3800.0],
        },
        out,
    )

    assert out.exists()
    # Matplotlib produit toujours au moins quelques Ko ; on garde le seuil bas
    # pour rester robuste aux variations de backend / DPI.
    assert out.stat().st_size > 1000


def test_save_latency_comparison_png_handles_single_backend(tmp_path: Path) -> None:
    from scripts.eval.plotting import save_latency_comparison_png

    out = tmp_path / "lat.png"
    save_latency_comparison_png({"gemma3:4b": [1.0, 2.0, 3.0]}, out)

    assert out.exists()
