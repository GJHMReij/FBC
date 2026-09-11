"""
Combine Model 1/2/3's ROC curves into a single comparison figure.

Run this AFTER running whichever of pe_model1_cbc_rf.py / pe_model2_cbc_diff_rf.py
/ pe_model3_cbc_diff_ddimer_rf.py you want compared -- each of those scripts
saves its own ROC curve data (model1_roc_data.json / model2_roc_data.json /
model3_roc_data.json) as a side effect of its normal run. This script doesn't
rerun any models; it only reads whichever of those JSON files already exist
(so it works fine with just 1 or 2 models run so far, not necessarily all 3)
and plots them together.

Run with the project venv:
    .venv/bin/python combine_roc_curves.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROC_DATA_FILES = [
    SCRIPT_DIR / "model1_roc_data.json",
    SCRIPT_DIR / "model2_roc_data.json",
    SCRIPT_DIR / "model3_roc_data.json",
]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main():
    fig, ax = plt.subplots(figsize=(7, 7))

    n_found = 0
    for path, color in zip(ROC_DATA_FILES, COLORS):
        if not path.exists():
            print(f"Skipping {path.name} (not found -- run that model first if you want it included)")
            continue
        with open(path) as f:
            data = json.load(f)
        ax.plot(data["fpr"], data["tpr"], color=color,
                 label=f"{data['label']} (AUC={data['auc']:.3f})")
        n_found += 1

    if n_found == 0:
        print("No model ROC data files found -- run at least one model script first.")
        return

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Model 1 vs 2 vs 3 - ROC curve comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = SCRIPT_DIR / "combined_roc_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nCombined ROC curve ({n_found} model(s)) saved to: {out_path}")


if __name__ == "__main__":
    main()
