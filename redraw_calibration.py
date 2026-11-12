"""
Redraw the calibration plots of Models 1-4 so that the numbers in the plot are the
Rubin-pooled slope and intercept (with 95% CI), the same numbers as in the tables and slides.
No model is retrained: it reads files that the model runs already wrote to results_temporal/:

    model{n}_predictions.csv   scan-level rows (stay on MyDRE; never leave it)
    model{n}_results.json      pooled test results (slope, intercept with CI)

and writes only figures:

    results_temporal/model{n}_calibration_curve_pooled.png

Run on MyDRE after the model runs:   python redraw_calibration.py      (or F5 in Spyder)
Optional:                            python redraw_calibration.py --results-dir <folder>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

TITLES = {1: "Model 1: CBC", 2: "Model 2: CBC + Diff", 3: "Model 3: CBC + Diff + D-dimer",
          4: "Model 4: D-dimer, age, sex, creat."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results_temporal"))
    args = ap.parse_args()
    res = Path(args.results_dir)
    plt.rcParams.update({"font.size": 15})
    for n, title in TITLES.items():
        pred_path, js_path = res / f"model{n}_predictions.csv", res / f"model{n}_results.json"
        if not pred_path.exists() or not js_path.exists():
            print(f"Model {n}: predictions or results.json not found, skipped")
            continue
        df = pd.read_csv(pred_path)
        df = df[df["set"] == "test"]
        y, p = df["y"].to_numpy().astype(int), df["p_ensemble"].to_numpy()
        with open(js_path) as f:
            pooled = json.load(f)["test_pooled"]
        sl, ic = pooled["slope"], pooled["intercept"]

        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        xs = np.linspace(0.001, 0.999, 200)
        line = 1 / (1 + np.exp(-(ic["est"] + sl["est"] * np.log(xs / (1 - xs)))))

        fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]})
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
        ax.plot(xs, line, color="tab:orange", lw=2,
                label=(f"Slope {sl['est']:.3f} ({sl['lo']:.2f} to {sl['hi']:.2f})\n"
                       f"Intercept {ic['est']:.3f} ({ic['lo']:.2f} to {ic['hi']:.2f})"))
        ax.plot(pred, obs, marker="o", ms=8, linestyle="none", color="tab:blue",
                label="Deciles of predicted risk")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=13)
        axh.hist(p, bins=np.linspace(0, 1, 51), color="tab:grey")
        axh.set_xlabel("Predicted probability")
        axh.set_ylabel("Scans")
        fig.tight_layout()
        out = res / f"model{n}_calibration_curve_pooled.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Model {n}: slope {sl['est']:.3f}, intercept {ic['est']:.3f} -> {out.name}")


if __name__ == "__main__":
    main()
