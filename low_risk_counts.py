"""
How many test scans get a (recalibrated) predicted PE probability of at most 1/2/3/5%?
For each cut-off: number of scans, share of the test set, PE among them, observed NPV.
Reads model{n}_predictions.csv (scan-level, stays on MyDRE); prints and writes only
aggregate counts to low_risk_counts.csv.

    python low_risk_counts.py [results_folder]    (default: results_temporal_outcome_imp; or F5 in Spyder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / (
    sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results_temporal_outcome_imp")
CUTS = [0.01, 0.02, 0.03, 0.05]

rows = []
for n in (1, 2, 3, 4):
    path = RESULTS / f"model{n}_predictions.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    for part in ("test", "train"):
        d = df[df["set"] == part]
        y, p = d["y"].to_numpy(), d["p_ensemble"].to_numpy()
        for c in CUTS:
            low = p <= c
            k = int(low.sum())
            pe = int(y[low].sum())
            rows.append({"model": n, "set": part, "max_probability": c, "n_scans_total": len(p),
                         "n_below": k, "pct_below": round(100 * k / len(p), 2), "pe_among_them": pe,
                         "observed_npv": round(1 - pe / k, 3) if k else np.nan})
    print(f"Model {n}: min predicted probability on test = {df[df['set'] == 'test']['p_ensemble'].min():.3f}, "
          f"median = {df[df['set'] == 'test']['p_ensemble'].median():.3f}")

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "low_risk_counts.csv", index=False)
print(out[out["set"] == "test"].to_string(index=False))
print(f"\nWritten to {RESULTS / 'low_risk_counts.csv'}")
