"""
Ensemble version of the test-set tables (one model = mean of the 10 forests, one cut-off from the
training set, applied once to the test set; 95% CI = patient-level bootstrap percentile).
Reads model{n}_results.json (written by the pipeline: no re-run needed) and prints/writes
sensitivity, specificity, PPV, NPV, efficiency for the 97/98/99% sensitivity targets and NPV >= 98%.

    python ensemble_table.py [results_folder]    (default: results_temporal_outcome_imp; or F5 in Spyder)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / (
    sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results_temporal_outcome_imp")
METRICS = ["sensitivity", "specificity", "ppv", "npv", "efficiency"]
TARGETS = [("97%", "A0.97"), ("98%", "A0.98"), ("99%", "A0.99"), ("NPV 98%", "Anpv")]

rows = []
for n in (1, 2, 3, 4):
    path = RESULTS / f"model{n}_results.json"
    if not path.exists():
        continue
    with open(path) as f:
        res = json.load(f)
    ens = res.get("ensemble_test", {})
    thr = res.get("ensemble_thresholds", {})
    for label, pre in TARGETS:
        row = {"model": n, "target": label}
        for m in METRICS:
            d = ens.get(f"{pre}_{m}")
            if d is None:
                row[m] = "?"
            elif d["est"] is None or np.isnan(d["est"]):
                row[m] = "not defined"
            else:
                row[m] = f"{d['est']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
        rows.append(row)

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40)
print(out.to_string(index=False))
out.to_csv(RESULTS / "ensemble_table.csv", index=False)
print(f"\nWritten to {RESULTS / 'ensemble_table.csv'}")
