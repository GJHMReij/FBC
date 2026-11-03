"""
Combined figures for all four models (ROC, calibration, scans saved), read from
results_temporal/. Run on MyDRE after the model runs; only figures are written
(model{n}_predictions.csv holds scan-level rows and never leaves MyDRE).

    python combine_figures.py            (or press F5 in Spyder)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results_temporal"
MODELS = {1: "Model 1 (CBC)", 2: "Model 2 (+DIFF)", 3: "Model 3 (+D-dimer)",
          4: "Model 4 (D-dimer, age, sex, creatinine)"}
COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}   # colour-blind safe
N_BINS = 10


def test_predictions(n):
    path = RESULTS / f"model{n}_predictions.csv"
    df = pd.read_csv(path)
    df = df[df["set"] == "test"]
    return df["y"].to_numpy(), df["p_ensemble"].to_numpy()


def roc_figure(available):
    fig, ax = plt.subplots(figsize=(7, 7))
    for n in available:
        with open(RESULTS / f"model{n}_roc_data.json") as f:
            d = json.load(f)
        ax.plot(d["fpr"], d["tpr"], color=COLORS[n], lw=2, label=f"{MODELS[n]} (AUC {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curves, hold-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "combined_roc.png", dpi=200)
    plt.close(fig)


def calibration_figure(available):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for n in available:
        y, p = test_predictions(n)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
        b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs = [p[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
        ys = [y[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
        ax.plot(xs, ys, "o-", color=COLORS[n], lw=1.8, ms=5, label=MODELS[n])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of PE")
    ax.set_ylabel("Observed proportion with PE")
    ax.set_title("Calibration, hold-out test set (deciles of predicted risk)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "combined_calibration.png", dpi=200)
    plt.close(fig)


def scans_saved_figure(available):
    targets = ["97%", "98%", "99%", "NPV 98%"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / len(available)
    for i, n in enumerate(available):
        t = pd.read_csv(RESULTS / f"model{n}_per1000_from_training.csv", index_col=0)
        vals = [float(t.loc[k, "scans saved (no CTPA)"]) if k in t.index else np.nan for k in targets]
        pos = np.arange(len(targets)) + (i - (len(available) - 1) / 2) * width
        bars = ax.bar(pos, vals, width, color=COLORS[n], label=MODELS[n])
        for r, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(["Sens 97%", "Sens 98%", "Sens 99%", "NPV 98%"])
    ax.set_ylabel("CTPA scans saved per 1,000")
    ax.set_title("Scans saved at thresholds set on the training set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "combined_scans_saved.png", dpi=200)
    plt.close(fig)


def main():
    avail_roc = [n for n in MODELS if (RESULTS / f"model{n}_roc_data.json").exists()]
    avail_pred = [n for n in MODELS if (RESULTS / f"model{n}_predictions.csv").exists()]
    avail_tab = [n for n in MODELS if (RESULTS / f"model{n}_per1000_from_training.csv").exists()]
    print(f"Models found: ROC {avail_roc}, predictions {avail_pred}, tables {avail_tab}")
    if avail_roc:
        roc_figure(avail_roc)
    if avail_pred:
        calibration_figure(avail_pred)
    if avail_tab:
        scans_saved_figure(avail_tab)
    print(f"Figures written to {RESULTS}: combined_roc.png, combined_calibration.png, combined_scans_saved.png")


if __name__ == "__main__":
    main()
