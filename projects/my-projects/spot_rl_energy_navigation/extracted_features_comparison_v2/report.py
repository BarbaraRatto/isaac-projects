"""Render plots and a Markdown report from measured comparisons; no simulator."""
from __future__ import annotations

import argparse
import csv
import json
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from common import local_path


def heatmap(matrix, labels, path, title, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=25)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white",
                    bbox={"facecolor": "black", "alpha": .4, "edgecolor": "none"})
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    directory = local_path(args.results)
    summary = json.loads((directory / "summary.json").read_text())
    text = ["# Confronto feature DINOv2", "", "Risultati misurati sui dati acquisiti, senza training RL.", "",
            "| Rappresentazione | Balanced accuracy | Intervallo bootstrap per pose (95%) |",
            "| --- | ---: | --- |"]
    complete = {name: result for name, result in summary["results"].items() if result.get("status") == "complete"}
    for representation, result in complete.items():
        labels = result["labels"]
        probe = result["linear_probe"]
        interval = probe["group_bootstrap_95_interval"]
        interval_text = f"{interval[0]:.1%} – {interval[1]:.1%}" if interval else "n.d."
        text.append(f"| {representation} | {probe['balanced_accuracy']:.1%} | {interval_text} |")
        heatmap(np.asarray(result["mean_cosine_matrix"]), labels, directory / f"{representation}_similarity.png",
                f"{representation}: similarità coseno media", -1, 1)
        cm = np.asarray(probe["confusion_matrix"], dtype=float)
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
        heatmap(cm, labels, directory / f"{representation}_confusion.png",
                f"{representation}: confusione (riga vera, colonna predetta)", 0, 1)
        with np.load(directory / f"{representation}_plot_samples.npz", allow_pickle=False) as data:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(data["features"])
            fig, ax = plt.subplots(figsize=(7, 5))
            for label in labels:
                mask = data["labels"] == label
                ax.scatter(xy[mask, 0], xy[mask, 1], s=9, alpha=.35, label=label)
            ax.legend()
            ax.set_title(f"{representation}: PCA solo illustrativa ({pca.explained_variance_ratio_.sum():.1%} varianza)")
            fig.tight_layout()
            fig.savefig(directory / f"{representation}_pca.png", dpi=160)
            plt.close(fig)
        with (directory / f"{representation}_cosine_samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        keys = list(dict.fromkeys((r["terrain_a"], r["terrain_b"]) for r in rows))
        fig, ax = plt.subplots(figsize=(max(8, len(keys)*.5), 6))
        values = [[float(r["cosine"]) for r in rows if (r["terrain_a"], r["terrain_b"]) == pair] for pair in keys]
        ax.boxplot(values, tick_labels=[f"{a}/{b}" for a, b in keys], showfliers=False)
        ax.set_ylabel("Similarità coseno")
        ax.tick_params(axis="x", rotation=65, labelsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"{representation}_distributions.png", dpi=160)
        plt.close(fig)
    text += ["", "Riferimento casuale della balanced accuracy: **14,3%** (sette classi).", "",
             "## Tutte le coppie", "", "| Rappresentazione | Coppia | Coseno stesso terreno | Coseno tra terreni | Differenza | AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for representation, result in complete.items():
        for pair in result["pair_results"]:
            text.append(f"| {representation} | {pair['a']} / {pair['b']} | {pair['within_cosine']['mean']:.3f} | "
                        f"{pair['between_cosine']['mean']:.3f} | {pair['within_minus_between']:.3f} | {pair['same_vs_different_auc']:.3f} |")
    for representation in complete:
        text += ["", f"### {representation}", "", f"![Similarità]({representation}_similarity.png)", "",
                 f"![Distribuzioni]({representation}_distributions.png)", "", f"![Confusione]({representation}_confusion.png)", "",
                 f"![PCA]({representation}_pca.png)", ""]
    text += ["## Suddivisione dei dati", ""]
    for representation, result in complete.items():
        text += [f"- {representation}: {result['linear_probe']['split_method']}"]
    for representation, result in summary["results"].items():
        if result.get("status") != "complete":
            text += [f"- **{representation}: copertura insufficiente**, terreni mancanti: {result.get('missing_labels')}. Nessuna accuratezza a sette classi calcolata."]
    text += ["", "## Interpretazione e limiti", "",
             "Una similarità più alta entro lo stesso terreno è un segnale favorevole. AUC=0,5 indica nessuna separazione mediante questa misura.", "",
             "La diagonale confronta viste diverse dello stesso terreno: non è impostata artificialmente a 1. PCA in 2D non prova né esclude separabilità in 384D.", "",
             "Il pilota separa gruppi di pose/percorsi; la raccolta completa separa anche i tasselli donatori. Lo stesso frame o episodio non può comparire nei due insiemi. L'intervallo è indicativo con pochi percorsi.", "",
             "Feature distinguibili non dimostrano che una policy RL le usi né che predicano il consumo energetico.", ""]
    text += [f"- {item}" for item in summary["limitations"]]
    (directory / "report.md").write_text("\n".join(text) + "\n")
    print(directory / "report.md")


if __name__ == "__main__":
    main()
