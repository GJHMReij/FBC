"""
D-dimer alone as a comparison for Models 3 and 4 (no machine learning).

Same population and the same 80/20 temporal split as Model 3/4: scans with a
measured D-dimer. The D-dimer level itself is the score (higher = more likely
PE). Operating points:
  - thresholds for 97/98/99% sensitivity and NPV 98%, determined on the
    TRAINING set and applied unchanged to the test set (as for the models)
  - the conventional cut-off and the age-adjusted cut-off (age x 10 ug/L above 50)
Results go to results_temporal/ddimer_only_* (Word report, JSON, CSVs, ROC).
Run on MyDRE via run_ddimer.py (same arguments as pe_pipeline.py).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import pe_pipeline as pp
from pe_model2_cbc_diff_rf import AGE_COL, D_DIMER_VALUE_COL

METRICS = ["sensitivity", "specificity", "ppv", "npv", "efficiency", "accuracy", "f1"]


def ci(vals):
    v = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main():
    args = pp.parse_args()
    args.models = [3]                       # D-dimer population: scans without D-dimer are excluded before the split
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = out_dir / "ddimer_only"
    rng = np.random.default_rng(args.seed)

    df, y, features, _ = pp.load_and_prepare(args)
    patients = pp.patient_ids(df)
    is_test, cutoff_date, n_both, n_moved = pp.temporal_split(df["scan_date"], patients, args.test_fraction)
    pop = df[D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna()
    train_mask = (pop & ~is_test).to_numpy()
    test_mask = (pop & is_test).to_numpy()
    y_tr, y_te = y[train_mask].to_numpy(), y[test_mask].to_numpy()
    s_tr = df.loc[train_mask, D_DIMER_VALUE_COL].to_numpy(dtype=float)
    s_te = df.loc[test_mask, D_DIMER_VALUE_COL].to_numpy(dtype=float)
    age_te = df.loc[test_mask, AGE_COL].to_numpy(dtype=float)
    g_te = patients[test_mask].to_numpy()
    print(f"Training: {train_mask.sum()} scans ({y_tr.sum()} PE); hold-out test: {test_mask.sum()} scans ({y_te.sum()} PE)")

    med = float(np.median(s_tr))
    mg_l = med < 20          # mg/L FEU (cut-off 0.5) versus ug/L (cut-off 500)
    base, per_year = (0.5, 0.01) if mg_l else (500.0, 10.0)
    unit = "mg/L FEU" if mg_l else "ug/L"
    print(f"D-dimer in training: median {med:g}, 5-95% {np.percentile(s_tr, 5):g}-{np.percentile(s_tr, 95):g} "
          f"-> assumed unit {unit}; conventional cut-off {base:g}, age-adjusted {per_year:g} x age above 50")
    print("CHECK this against the units in your data dictionary before using the conventional cut-offs.")

    thr_age = np.where(age_te > 50, age_te * per_year, base)
    points = {}                              # label -> threshold (scalar or per-scan array on the test set)
    for t in pp.SENS_TARGETS:
        points[f"Sensitivity {int(t * 100)}%"] = pp.thr_for_sens(y_tr, s_tr, t)
    points[f"NPV {int(pp.NPV_TARGET * 100)}%"] = pp.thr_for_npv(y_tr, s_tr, pp.NPV_TARGET)
    points[f"Conventional ({base:g} {unit})"] = base
    points["Age-adjusted"] = thr_age

    draws = pp.make_draws(g_te, args.n_bootstrap, rng)
    prev = float(y_te.mean())
    rows, per1000, res = {}, {}, {}
    for label, thr in points.items():
        point = pp.metrics_np(y_te, s_te, thr)
        boots = [pp.metrics_np(y_te[i], s_te[i], thr if np.isscalar(thr) else thr[i]) for i in draws]
        rows[label] = {k: "{:.3f} [{:.3f}, {:.3f}]".format(point[k], *ci([b[k] for b in boots])) for k in METRICS}
        per1000[label] = {
            "PE per 1000 scans": round(1000 * prev),
            "scans saved (no CTPA)": round(1000 * point["efficiency"]),
            "of which without PE": round(1000 * point["specificity"] * (1 - prev)),
            "PE missed": round(1000 * prev * (1 - point["sensitivity"]), 1),
            "% of scans saved": f"{point['efficiency'] * 100:.1f}",
            "NPV": f"{point['npv']:.3f}",
        }
        res[label] = {k: {"est": float(point[k]), "lo": ci([b[k] for b in boots])[0],
                          "hi": ci([b[k] for b in boots])[1]} for k in METRICS}
        res[label]["threshold"] = None if not np.isscalar(thr) else float(thr)
    tbl = pd.DataFrame(rows).T
    tbl.index.name = "Cut-off"
    tbl1000 = pd.DataFrame(per1000).T
    tbl1000.index.name = "Cut-off"
    print("\nD-dimer alone, hold-out test set:\n" + tbl.to_string())
    print("\nPer 1000 scans:\n" + tbl1000.to_string())

    auc = float(roc_auc_score(y_te, s_te))
    auc_lo, auc_hi = ci([roc_auc_score(y_te[i], s_te[i]) for i in draws])
    print(f"\nAUC of the D-dimer level (test): {auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})")

    fpr, tpr, _ = roc_curve(y_te, s_te)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#CC79A7", lw=2, label=f"D-dimer alone (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("D-dimer alone, hold-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{pre}_roc.png", dpi=150)
    plt.close(fig)

    tbl.to_csv(f"{pre}_test_thresholds_from_training.csv")
    tbl1000.to_csv(f"{pre}_per1000.csv")
    with open(f"{pre}_roc_data.json", "w") as f:
        json.dump({"label": "D-dimer alone", "auc": auc, "fpr": [float(v) for v in fpr],
                   "tpr": [float(v) for v in tpr]}, f)
    with open(f"{pre}_results.json", "w") as f:
        json.dump({"label": "D-dimer alone", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                   "pe_train": int(y_tr.sum()), "pe_test": int(y_te.sum()),
                   "auc": {"est": auc, "lo": auc_lo, "hi": auc_hi}, "unit": unit,
                   "operating_points": res}, f, indent=1)

    try:
        from pe_report import build_report
        info = {
            "label": "D-dimer alone",
            "intro": "Comparison for Models 3 and 4: the D-dimer level itself is the score; no model is fitted.",
            "split": [
                f"Population: scans with a measured D-dimer ({int(pop.sum())} scans), same split as Models 3 and 4",
                f"Training: {int(train_mask.sum())} scans ({int(y_tr.sum())} PE, {y_tr.mean() * 100:.1f}%)",
                f"Hold-out test: {int(test_mask.sum())} scans ({int(y_te.sum())} PE, {y_te.mean() * 100:.1f}%)",
            ],
            "settings": [
                ("Score", "D-dimer level (higher = more likely PE)"),
                ("Unit (assumed)", unit),
                ("Thresholds", "97/98/99% sensitivity and NPV 98% from the training set, applied unchanged to the test set"),
                ("Fixed cut-offs", f"conventional {base:g} {unit}; age-adjusted {per_year:g} x age above 50 years"),
                ("Confidence intervals", f"patient-level bootstrap, {len(draws)} draws"),
            ],
            "test_metrics": [("AUC (test)", f"{auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})")],
            "tables": [
                ("Test set: cut-offs and metrics", tbl,
                 "Thresholds for the sensitivity and NPV targets come from the training set."),
                ("Scans saved and PE missed per 1000 scans", tbl1000,
                 "Scans saved = scans below the cut-off (efficiency x 1000); PE missed = (1 - sensitivity) x PE per 1000."),
            ],
            "importance": None,
            "images": [("ROC curve", f"{pre}_roc.png")],
        }
        print("Report written:", build_report(pre, info))
    except Exception as exc:                 # the results above are already saved
        print("WARNING: report not written:", exc)


if __name__ == "__main__":
    main()
