"""
NPV >= 98% cut-off on the ENSEMBLE (mean probability of the 10 forests): how many test scans
fall below the cut-off from the training set, how many of them have PE, and the resulting NPV.
Unlike the per-imputation pooling in the report, the ensemble is one model with one cut-off, so
the NPV is defined whenever at least one test scan falls below it. Only aggregate counts are
printed/written (model{n}_predictions.csv stays on MyDRE).

    python npv98_ensemble.py [results_folder]    (default: results_temporal_outcome_imp; or F5 in Spyder)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / (
    sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results_temporal_outcome_imp")

rows = []
for n in (1, 2, 3, 4):
    pj, pc = RESULTS / f"model{n}_results.json", RESULTS / f"model{n}_predictions.csv"
    if not (pj.exists() and pc.exists()):
        continue
    with open(pj) as f:
        res = json.load(f)
    thr = res["ensemble_thresholds"]["npv"]
    df = pd.read_csv(pc)
    t = df[df["set"] == "test"]
    y, p = t["y"].to_numpy(), t["p_ensemble"].to_numpy()
    below = p < thr                       # same rule as the pipeline: pred = p >= thr is "positive"
    k, pe = int(below.sum()), int(y[below].sum())
    ens = res.get("ensemble_test", {})
    e = ens.get("Anpv_npv")
    rows.append({
        "model": n, "cutoff_from_training": round(float(thr), 4), "test_scans": len(y),
        "n_below_cutoff": k, "pct_below": round(100 * k / len(y), 2), "pe_below": pe,
        "npv_test": round(1 - pe / k, 3) if k else "not defined (no scans below)",
        "npv_bootstrap_ci": (f"{e['lo']:.3f}-{e['hi']:.3f}" if e and not np.isnan(e["est"]) else ""),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(RESULTS / "npv98_ensemble.csv", index=False)
print(f"\nWritten to {RESULTS / 'npv98_ensemble.csv'}")
