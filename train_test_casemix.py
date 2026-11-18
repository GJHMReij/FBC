"""
Train vs test: prevalence and case-mix per model (no modelling, runs in seconds).

Uses the same selection and temporal split as pe_pipeline.py: Models 1+2 on all
scans, Models 3+4 on the scans with a measured D-dimer (excluded before the split).
For each population it prints and saves a table with training set, test set and the
standardised mean difference (SMD; |SMD| > 0.1 is marked with * = noticeable difference):
  n scans, PE, age, sex, hospital, patient class, scan period, creatinine, D-dimer.

Output: results_temporal/train_test_casemix.csv and .docx (.html if python-docx is missing).
Run in Spyder on MyDRE with F5, then copy-paste the printed block.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_pipeline as pp

INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"


def load(models, require_ddimer):
    args = argparse.Namespace(
        input_csv=Path(INPUT_CSV), outcome_correction=Path(OUTCOME_CORRECTION),
        models=models, require_ddimer=require_ddimer, fake_dates_for_testing=False,
    )
    with redirect_stdout(io.StringIO()):
        df, y, _, _ = pp.load_and_prepare(args)
        is_test, _, _, _ = pp.temporal_split(df["scan_date"], pp.patient_ids(df), 0.2)
    return df, y, np.asarray(is_test, dtype=bool)


def smd_binary(p1, p2):
    d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return 0.0 if d == 0 else (p1 - p2) / d


def smd_cont(a, b):
    a, b = a.dropna(), b.dropna()
    d = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return np.nan if (d == 0 or np.isnan(d)) else (a.mean() - b.mean()) / d


def fmt_smd(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return f"{s:+.2f}" + ("*" if abs(s) > 0.1 else "")


def cont_row(label, a, b):
    def f(x):
        x = x.dropna()
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        return f"{med:.1f} ({q1:.1f}-{q3:.1f})"
    return (label + ", median (IQR)", f(a), f(b), fmt_smd(smd_cont(a, b)))


def binary_row(label, a, b):
    p1, p2 = float(a.mean()), float(b.mean())
    return (label, f"{int(a.sum())} ({100 * p1:.1f}%)", f"{int(b.sum())} ({100 * p2:.1f}%)",
            fmt_smd(smd_binary(p1, p2)))


def cat_rows(title, a, b):
    rows = [(title, "", "", "")]
    a, b = a.astype(str).str.strip(), b.astype(str).str.strip()
    levels = sorted(set(a) | set(b), key=lambda v: -((a == v).sum() + (b == v).sum()))
    for v in levels:
        r = binary_row("   " + v, (a == v).astype(float), (b == v).astype(float))
        rows.append(r)
    return rows


def table(df, y, is_test):
    tr, te = df.loc[~is_test], df.loc[is_test]
    ytr, yte = y[~is_test], y[is_test]
    rows = [("Scans, n", f"{len(tr)}", f"{len(te)}", "")]
    rows.append(binary_row("PE", ytr.astype(float), yte.astype(float)))
    rows.append(("Scan period (first - last)",
                 f"{tr['scan_date'].min():%Y-%m} - {tr['scan_date'].max():%Y-%m}",
                 f"{te['scan_date'].min():%Y-%m} - {te['scan_date'].max():%Y-%m}", ""))
    rows.append(cont_row("Age (years)", tr[pp.AGE_COL], te[pp.AGE_COL]))
    rows.append(binary_row("Female", 1 - tr["sex_enc"], 1 - te["sex_enc"]))
    if pp.HOSPITAL_COL in df.columns:
        rows += cat_rows("Hospital", tr[pp.HOSPITAL_COL], te[pp.HOSPITAL_COL])
    if pp.CLASS_COL in df.columns:
        rows += cat_rows("Patient class", tr[pp.CLASS_COL].fillna("missing"),
                         te[pp.CLASS_COL].fillna("missing"))
    rows.append(cont_row("Creatinine", tr[pp.CREATININE_COL], te[pp.CREATININE_COL]))
    rows.append(("   creatinine missing", f"{100 * tr[pp.CREATININE_COL].isna().mean():.1f}%",
                 f"{100 * te[pp.CREATININE_COL].isna().mean():.1f}%", ""))
    rows.append(cont_row("D-dimer (measured values)", tr[pp.D_DIMER_VALUE_COL], te[pp.D_DIMER_VALUE_COL]))
    rows.append(("   D-dimer missing", f"{100 * tr[pp.D_DIMER_VALUE_COL].isna().mean():.1f}%",
                 f"{100 * te[pp.D_DIMER_VALUE_COL].isna().mean():.1f}%", ""))
    return rows


results = {}
for title, models, req in [("Models 1 and 2 (all scans)", [1], False),
                           ("Models 3 and 4 (scans with D-dimer)", [3], True)]:
    df, y, is_test = load(models, req)
    results[title] = table(df, y, is_test)

width = max(len(r[0]) for rows in results.values() for r in rows) + 2
all_lines = []
for title, rows in results.items():
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'':<{width}}{'Training':>22}{'Test':>22}{'SMD':>9}")
    for label, a, b, s in rows:
        print(f"{label:<{width}}{a:>22}{b:>22}{s:>9}")
        all_lines.append((title, label, a, b, s))
    print()
print("SMD = standardised mean difference (train - test); * = |SMD| > 0.1")

out_dir = Path(pp.REPO_ROOT) / "results_temporal"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(all_lines, columns=["population", "variable", "training", "test", "smd"]).to_csv(
    out_dir / "train_test_casemix.csv", index=False)
try:
    import docx
    doc = docx.Document()
    doc.add_heading("Train vs test: prevalence and case-mix", level=0)
    for title, rows in results.items():
        doc.add_heading(title, level=1)
        t = doc.add_table(rows=1, cols=4)
        t.style = "Light Grid Accent 1"
        for c, h in zip(t.rows[0].cells, ["", "Training", "Test", "SMD"]):
            c.text = h
        for label, a, b, s in rows:
            cells = t.add_row().cells
            for c, v in zip(cells, [label.replace("   ", "  "), a, b, s]):
                c.text = v
    doc.add_paragraph("SMD = standardised mean difference (training - test); * = |SMD| > 0.1.")
    path = out_dir / "train_test_casemix.docx"
    doc.save(str(path))
except ImportError:
    html = ["<html><head><meta charset='utf-8'></head><body><h1>Train vs test</h1>"]
    for title, rows in results.items():
        html.append(f"<h2>{title}</h2><table border=1><tr><th></th><th>Training</th><th>Test</th><th>SMD</th></tr>")
        html += [f"<tr><td>{l}</td><td>{a}</td><td>{b}</td><td>{s}</td></tr>" for l, a, b, s in rows]
        html.append("</table>")
    html.append("</body></html>")
    path = out_dir / "train_test_casemix.html"
    path.write_text("\n".join(html), encoding="utf-8")
print(f"Table written: {path}")
print("\nCopy everything above and paste it in the chat.")
