"""
Univariable AUC of every CBC and DIFF variable (plus age, creatinine, D-dimer for reference) for PE,
on the SAME temporal split as the main pipeline. Raw (not imputed) values, complete cases per variable.

Per variable: % missing, n, AUC in the training set and in the test set. The direction (higher value =
more PE, or lower value = more PE) is fixed on the TRAINING set and the same direction is used for the
test set, so a test AUC < 0.5 means the training signal did not hold. Test AUC has a patient-level
bootstrap 95% CI. Sorted by TRAINING AUC (so the ranking is not tuned on the test set).
Two populations: all scans (Models 1/2) and scans with a measured D-dimer (Models 3/4).
With ~140 variables a few will look good by chance: look at train AND test together.

Output: console top lists + univariable_auc.csv in --out-dir (feature-level aggregates, no patient data).

    python univariable_auc.py --models 1 --input-csv ... --outcome-correction ... --out-dir results_temporal_outcome_imp
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pe_pipeline as pp
from pe_model1_cbc_rf import AGE_COL, CREATININE_COL
from pe_model2_cbc_diff_rf import D_DIMER_VALUE_COL

N_BOOT = 200


def auc_ok(y, x):
    return len(y) > 20 and 0 < y.sum() < len(y) and np.unique(x).size > 1


def uni_auc(y, x):
    m = ~np.isnan(x)
    return (roc_auc_score(y[m], x[m]) if auc_ok(y[m], x[m]) else np.nan), int(m.sum())


def main():
    args = pp.parse_args()
    df, y, features, imputation_cols = pp.load_and_prepare(args)
    patients = pp.patient_ids(df)
    is_test, cutoff, _, _ = pp.temporal_split(df["scan_date"], patients, args.test_fraction)
    cbc = [f for f in features[1] if f not in ("sex_enc", AGE_COL, CREATININE_COL)]
    diff_only = [f for f in features[2] if f not in features[1]]
    groups = {f: "CBC" for f in cbc}
    groups.update({f: "DIFF" for f in diff_only})
    groups.update({AGE_COL: "ref", CREATININE_COL: "ref", D_DIMER_VALUE_COL: "ref"})
    rng = np.random.default_rng(args.seed + 7)

    rows = []
    for pop_name, pop in [("all_scans", pd.Series(True, index=df.index)),
                          ("ddimer_measured", df[D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna())]:
        tr, te = (pop & ~is_test).to_numpy(), (pop & is_test).to_numpy()
        ytr, yte = y[tr].to_numpy(), y[te].to_numpy()
        gte = patients[te].to_numpy()
        draws = pp.make_draws(gte, N_BOOT, rng)
        print(f"\n{pop_name}: training {tr.sum()} scans, test {te.sum()} scans")
        for f, grp in groups.items():
            if f not in df.columns:
                continue
            xtr = pd.to_numeric(df.loc[tr, f], errors="coerce").to_numpy(dtype=float)
            xte = pd.to_numeric(df.loc[te, f], errors="coerce").to_numpy(dtype=float)
            a_tr, n_tr = uni_auc(ytr, xtr)
            if np.isnan(a_tr):
                continue
            sign = 1.0 if a_tr >= 0.5 else -1.0           # direction fixed on the training set
            a_te, n_te = uni_auc(yte, sign * xte)
            boots = []
            for idx in draws:
                a, _ = uni_auc(yte[idx], sign * xte[idx])
                if not np.isnan(a):
                    boots.append(a)
            rows.append({"population": pop_name, "feature": f, "group": grp,
                         "direction": "higher = more PE" if sign > 0 else "lower = more PE",
                         "pct_missing_train": round(100 * np.isnan(xtr).mean(), 1),
                         "n_train": n_tr, "auc_train": round(max(a_tr, 1 - a_tr), 3),
                         "n_test": n_te, "auc_test": round(a_te, 3) if not np.isnan(a_te) else np.nan,
                         "test_ci_lo": round(float(np.percentile(boots, 2.5)), 3) if boots else np.nan,
                         "test_ci_hi": round(float(np.percentile(boots, 97.5)), 3) if boots else np.nan})
    out = pd.DataFrame(rows).sort_values(["population", "auc_train"], ascending=[True, False])
    path = args.out_dir / "univariable_auc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    cols = ["feature", "group", "direction", "pct_missing_train", "auc_train", "auc_test", "test_ci_lo", "test_ci_hi"]
    for pop_name, sub in out.groupby("population"):
        print(f"\n=== {pop_name}: top 20 by training AUC ===")
        print(sub[cols].head(20).to_string(index=False))
        ref = sub[sub["group"] == "ref"]
        print(f"\n--- reference ({pop_name}) ---")
        print(ref[cols].to_string(index=False))
        print(f"\nVariables with test AUC (in training direction) >= 0.55: {int((sub['auc_test'] >= 0.55).sum())} of {len(sub)}")
    print(f"\nWritten: {path}\nPaste the top lists here (aggregates only).")


if __name__ == "__main__":
    main()
