"""Figure 2: row-normalised confusion matrices for the three VLMs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/openrouter_study_complete"
OUT_DIR = RESULTS / "figures"

CLASSES = ["CNV", "DRUSEN", "DME", "CSR", "MH", "NORMAL"]
TICK_LABELS = ["CNV", "Drusen", "DME", "CSR", "MH", "Normal"]
MODELS = [
    ("a", "Gemini-3-Flash", "google_gemini-3-flash-preview_predictions.csv"),
    ("b", "GPT-5.4-mini", "openai_gpt-5.4-mini_predictions.csv"),
    ("c", "Llama-3.2-11B-Vision-Instruct", "meta-llama_llama-3.2-11b-vision-instruct_predictions.csv"),
]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})


def row_normalised(csv_path):
    df = pd.read_csv(csv_path)
    counts = confusion_matrix(df["true_label"], df["predicted_label"], labels=CLASSES)
    assert counts.sum() == len(df) == 919
    return counts / counts.sum(axis=1, keepdims=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    for ax, (letter, name, csv_name) in zip(axes, MODELS):
        cm = row_normalised(RESULTS / csv_name)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(CLASSES)), TICK_LABELS, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(CLASSES)), TICK_LABELS)
        ax.set_xlabel("Model prediction")
        ax.set_title(f"$\\bf{{{letter}}}$  {name}", loc="left", fontsize=8)
        ax.tick_params(length=2, width=0.5)
    axes[0].set_ylabel("Reference diagnosis")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Proportion of reference class")
    cbar.ax.tick_params(length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(OUT_DIR / "figure2_confusion_matrices.pdf")
    fig.savefig(OUT_DIR / "figure2_confusion_matrices.png", dpi=600)
    fig.savefig(OUT_DIR / "figure2_confusion_matrices.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("Figure 2 written to", OUT_DIR)


if __name__ == "__main__":
    main()
