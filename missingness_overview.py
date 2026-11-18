"""
How much is missing, and what is imputed? (no modelling, runs in seconds)

Uses the same selection as pe_pipeline.py. Prints, for the population of
Models 1 and 2 (all scans) and of Models 3 and 4 (scans with a D-dimer):
  - % missing for age, sex, creatinine, D-dimer (value and assay), CBC+DIFF as a group
  - the CBC+DIFF parameters with the most missing values
  - how many scans have at least one missing value among the model's features
Also writes results_temporal/missingness.csv (one row per variable and population).

Run in Spyder on MyDRE with F5, then copy-paste the printed block.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_pipeline as pp

INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"

args = argparse.Namespace(
    input_csv=Path(INPUT_CSV), outcome_correction=Path(OUTCOME_CORRECTION),
    models=[1], require_ddimer=False, fake_dates_for_testing=False,
)
with redirect_stdout(io.StringIO()):
    df, y, features, imputation_cols = pp.load_and_prepare(args)

dd_pop = df[pp.D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna()
populations = {"Models 1+2 (all scans)": pd.Series(True, index=df.index),
               "Models 3+4 (D-dimer measured)": dd_pop}

named = {"age": pp.AGE_COL, "sex": "sex_enc", "creatinine": pp.CREATININE_COL,
         "D-dimer value": pp.D_DIMER_VALUE_COL, "D-dimer assay": "D_dimer_assay_enc"}
blood = [c for c in features[2] if c not in ("sex_enc", pp.AGE_COL, pp.CREATININE_COL)]
cbc_only = [c for c in features[1] if c not in ("sex_enc", pp.AGE_COL, pp.CREATININE_COL)]

rows = []
for pop_name, mask in populations.items():
    d = df.loc[mask]
    print("=" * 66)
    print(f"{pop_name}: {len(d)} scans")
    print("=" * 66)
    for label, col in named.items():
        pct = 100 * d[col].isna().mean()
        print(f"  {label:<16} missing {pct:5.1f}%   ({int(d[col].isna().sum())} scans)")
        rows.append((pop_name, label, col, round(pct, 2)))
    for label, cols in [("CBC parameters", cbc_only), ("CBC + DIFF parameters", blood)]:
        pcts = 100 * d[cols].isna().mean()
        print(f"  {label}: {len(cols)} variables; missing per variable: "
              f"median {pcts.median():.1f}%, range {pcts.min():.1f}-{pcts.max():.1f}%")
        for lo, hi in [(0, 0), (0, 5), (5, 20), (20, 100)]:
            n = int(((pcts > lo) & (pcts <= hi)).sum()) if lo or hi else int((pcts == 0).sum())
            lab = "0%" if (lo, hi) == (0, 0) else f"{lo}-{hi}%"
            print(f"      {lab:>8}: {n} variables")
    for c, pct in (100 * d[blood].isna().mean()).items():
        rows.append((pop_name, "blood parameter", c, round(float(pct), 2)))
    print("  Highest missing among CBC+DIFF parameters:")
    for c, pct in (100 * d[blood].isna().mean()).sort_values(ascending=False).head(10).items():
        print(f"      {c:<24} {pct:5.1f}%")
    for m in (1, 2, 3, 4):
        if m in (3, 4) and pop_name.startswith("Models 1"):
            continue
        if m in (1, 2) and pop_name.startswith("Models 3"):
            continue
        any_na = d[features[m]].isna().any(axis=1).mean() * 100
        print(f"  Model {m}: {any_na:.1f}% of scans have at least one missing feature")
    print()

out_dir = Path(pp.REPO_ROOT) / "results_temporal"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["population", "variable", "column", "pct_missing"]).to_csv(
    out_dir / "missingness.csv", index=False)
print(f"Table written: {out_dir / 'missingness.csv'}")
print("\nCopy everything above and paste it in the chat.")
