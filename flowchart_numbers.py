"""
Prints the numbers needed for the selection flowchart (no modelling, runs in
seconds). Runs the same selection and split as pe_pipeline.py for Model 1/2 and
for Model 3 (scans without D-dimer excluded before the split).

Run in Spyder on MyDRE with F5, then copy-paste the printed block.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_pipeline as pp

INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
FAKE_DATES = False   # True only for a test on the dummy data on the Mac


def numbers(models, require_ddimer):
    args = argparse.Namespace(
        input_csv=Path(INPUT_CSV), outcome_correction=Path(OUTCOME_CORRECTION),
        models=models, require_ddimer=require_ddimer, fake_dates_for_testing=FAKE_DATES,
    )
    with redirect_stdout(io.StringIO()):   # keep the output short
        df, y, _, _ = pp.load_and_prepare(args)
        is_test, cutoff, n_both, n_late = pp.temporal_split(df["scan_date"], pp.patient_ids(df), 0.2)
    train, test = ~is_test, is_test
    out = dict(df.attrs["selection"])
    out.update({
        "n_train": int(train.sum()), "pe_train": int(y[train].sum()),
        "n_test": int(test.sum()), "pe_test": int(y[test].sum()),
        "test_share_pct": round(100 * test.mean(), 2),
        "patients_in_both_periods": int(n_both),
        "pe_total": int(y.sum()), "pe_pct": round(100 * y.mean(), 1),
        "first_scan_date": str(df["scan_date"].min().date()),
        "last_scan_date": str(df["scan_date"].max().date()),
        "n_patients": int(pp.patient_ids(df).nunique()),
    })
    if pp.HOSPITAL_COL in df.columns:
        out["hospital_train"] = df.loc[train, pp.HOSPITAL_COL].value_counts().to_dict()
        out["hospital_test"] = df.loc[test, pp.HOSPITAL_COL].value_counts().to_dict()
    return out


print("=" * 60)
print("MODELS 1 AND 2 (all scans)")
print("=" * 60)
for k, v in numbers([1], False).items():
    print(f"{k}: {v}")
print()
print("=" * 60)
print("MODEL 3 (scans without D-dimer excluded before the split)")
print("=" * 60)
for k, v in numbers([3], True).items():
    print(f"{k}: {v}")
print("\nCopy everything above and paste it in the chat.")
