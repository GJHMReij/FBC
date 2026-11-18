"""
Checks what the first exclusion step of the selection flowchart really is.

Question: is the "-512" on the flowchart the outpatient scans (Patientklasse =
Poliklinisch), or scans < 3 months after the previous scan of the same patient?
The pipeline (pe_pipeline.py) only excludes outpatients; it has no
"< 3 months after previous scan" step.

Prints, for the file as read by the pipeline (after the outcome correction):
  - number of scans, number with a Ja/Nee outcome
  - number of outpatient scans
  - number of scans < 3 months (and < 1 month) after the previous scan of the same
    patient, split by outpatient / not outpatient
  - what is left after removing outpatients only

Run in Spyder on MyDRE with F5, then copy-paste the printed block. No modelling.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_pipeline as pp

INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"

df = pd.read_csv(INPUT_CSV, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Rows in file: {len(df)}")
df = pp.apply_outcome_correction(df, Path(OUTCOME_CORRECTION)).reset_index(drop=True)
print(f"Rows after outcome correction: {len(df)}")

ok = df[pp.OUTCOME_COL].isin(["Ja", "Nee"])
print(f"Rows without Ja/Nee outcome: {int((~ok).sum())}")
df = df.loc[ok].reset_index(drop=True)
print(f"Rows with Ja/Nee outcome: {len(df)}")

# ---- outpatients
if pp.CLASS_COL in df.columns:
    print("\nPatientklasse values:")
    print(df[pp.CLASS_COL].astype(str).str.strip().value_counts(dropna=False).to_string())
    poli = df[pp.CLASS_COL].astype(str).str.match(pp.EXCLUDE_CLASS_PATTERN, case=False)
else:
    print(f"\nWARNING: '{pp.CLASS_COL}' not found")
    poli = pd.Series(False, index=df.index)
print(f"\nOutpatient scans: {int(poli.sum())}")

# ---- interval to the previous scan of the same patient (all scans in the file)
parsed, n_ok, fmt = pp.parse_dates(df[pp.DATE_COL])
d = pd.DataFrame({"pid": df[pp.PATIENT_COL].astype(str), "date": parsed, "poli": poli})
d = d.sort_values(["pid", "date"])
d["gap_days"] = d.groupby("pid")["date"].diff().dt.days
d = d.loc[df.index]  # back to original order is not needed; counts only

print(f"Valid dates: {n_ok} of {len(df)} (format {fmt})")
print(f"Scans with an earlier scan of the same patient: {int(d['gap_days'].notna().sum())}")
for days, label in [(90, "3 months (90 d)"), (30, "1 month (30 d)")]:
    short = d["gap_days"] < days
    print(f"\n< {label} after the previous scan: {int(short.sum())}")
    print(f"   of which outpatient:     {int((short & d['poli']).sum())}")
    print(f"   of which not outpatient: {int((short & ~d['poli']).sum())}")

print(f"\nLeft after removing outpatients only: {int((~poli).sum())}")
print("Compare with the flowchart: 8,202 - 512 = 7,690")
print("\nCopy everything above and paste it in the chat.")
