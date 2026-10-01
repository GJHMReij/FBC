"""
Quick comparison baseline: simple logistic regression using Model 3's full
feature set (147 features, including D-dimer), validated via fast 5-fold
cross-validation per imputation -- NOT the full 500-resample bootstrap
procedure used for the Random Forest models. This is deliberately a quicker,
less rigorous check (a few minutes of modeling, though MICE itself still
takes the same time as Model 3's MICE step), meant to give a first
impression of how a traditional logistic regression stacks up against the
Random Forest models, not a publication-grade result.

Reuses the exact same data-loading/cleaning/outcome-correction/MICE pipeline
as pe_model3_cbc_diff_ddimer_rf.py so the comparison is apples-to-apples
(same patients, same features, same imputations).

Run with the project venv:
    .venv/bin/python compare_logreg.py --input-csv /path/to/real_cohort.csv \
        --outcome-correction /path/to/correction.xlsx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pe_model1_cbc_rf import (
    AGE_COL, CREATININE_COL, DEFAULT_INPUT_CSV, NOT_ASSESSABLE_VALUE,
    ORDER_ID_COL, OUTCOME_COL, OUTCOME_CORRECTION_COL, PROTECTED_COLS,
    REPO_ROOT, SEX_COL,
    apply_outcome_correction, calibration_slope_intercept, clean_data,
    read_dictionary, build_feature_channel_map,
)
from pe_model2_cbc_diff_rf import (
    D_DIMER_ASSAY_COL, D_DIMER_ASSAY_MAP, D_DIMER_VALUE_COL, N_IMPUTATIONS,
    build_imputation_frame, get_cbc_diff_features, run_mice,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="Path to the cohort CSV (defaults to the synthetic dummy dataset)",
    )
    parser.add_argument(
        "--n-imputations", type=int, default=N_IMPUTATIONS,
        help=f"Number of MICE imputation sets (default: {N_IMPUTATIONS}, matching Model 3)",
    )
    parser.add_argument(
        "--outcome-correction", type=Path, default=None,
        help="Path to an Excel/CSV file with the definitive, manually-reviewed PE "
             f"outcome (column '{OUTCOME_CORRECTION_COL}', joined on '{ORDER_ID_COL}'), "
             f"overriding '{OUTCOME_COL}'. Rows marked '{NOT_ASSESSABLE_VALUE}' or "
             "without a match are dropped.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = args.input_csv

    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Shape: {df.shape}")

    if args.outcome_correction is not None:
        print(f"\nApplying outcome correction from {args.outcome_correction} ...")
        df = apply_outcome_correction(df, args.outcome_correction)

    dictionary_df = read_dictionary()
    feature_channel_map = build_feature_channel_map(dictionary_df)
    cbc_diff_features = get_cbc_diff_features(feature_channel_map)
    cbc_diff_features = [c for c in cbc_diff_features if c in df.columns]
    df, cbc_diff_features = clean_data(df, cbc_diff_features)

    numeric_features = [
        c for c in cbc_diff_features
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]

    sex_map = {"Man": 1.0, "Vrouw": 0.0}
    df["Geslacht_enc"] = df[SEX_COL].map(sex_map)
    df["D_dimer_assay_enc"] = df[D_DIMER_ASSAY_COL].map(D_DIMER_ASSAY_MAP)
    y = (df[OUTCOME_COL] == "Ja").astype(int)

    model3_features = (
        ["Geslacht_enc", AGE_COL, CREATININE_COL, D_DIMER_VALUE_COL, "D_dimer_assay_enc"]
        + numeric_features
    )
    print(f"\nFeature count (same as Model 3): {len(model3_features)}")

    # Same core-covariate drop as Model 3 -- D-dimer is a required covariate
    # here too, not imputed.
    PROTECTED_COLS.update({D_DIMER_VALUE_COL, "D_dimer_assay_enc"})
    core_cols = ["Geslacht_enc", AGE_COL, CREATININE_COL, D_DIMER_VALUE_COL, "D_dimer_assay_enc"]
    complete_core_mask = df[core_cols].notna().all(axis=1) & y.notna()
    n_dropped = (~complete_core_mask).sum()
    print(f"Dropping {n_dropped} rows with missing core covariates or outcome "
          f"({n_dropped / len(df) * 100:.1f}%)")
    df = df.loc[complete_core_mask].reset_index(drop=True)
    y = y.loc[complete_core_mask].reset_index(drop=True)
    print(f"N remaining: {len(df)} (events={y.sum()})")

    imputation_frame = build_imputation_frame(df, y, model3_features)
    print(f"\nRunning MICE ({args.n_imputations} imputations) -- this is the slow "
          f"part, same cost as Model 3's MICE step...")
    imputed_datasets = run_mice(imputation_frame, model3_features, args.n_imputations)

    # Per imputation: fit logistic regression, validate via fast 5-fold CV
    # (out-of-fold predictions -- genuinely held-out, like the RF models'
    # OOB predictions, just via a different, faster mechanism). No repeated
    # bootstrap resampling here, so this is far quicker than Model 3's
    # validation, at the cost of a less precise uncertainty estimate.
    all_oof_preds = []
    all_coefs = []
    for m, X_m in enumerate(imputed_datasets):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_m[model3_features])

        lr = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, class_weight="balanced", random_state=42,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + m)
        oof_pred = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        all_oof_preds.append(oof_pred)

        # Also fit on all data (for coefficients/feature ranking, imputation 1 only)
        if m == 0:
            lr.fit(X_scaled, y)
            all_coefs = pd.Series(lr.coef_[0], index=model3_features)

        auc_m = roc_auc_score(y, oof_pred)
        print(f"  Imputation {m + 1}/{args.n_imputations}: 5-fold CV AUC = {auc_m:.3f}")

    # Pool across imputations: simple average of the out-of-fold predictions
    # (consistent with how the RF models' OOB predictions are pooled).
    pooled_pred = np.mean(all_oof_preds, axis=0)
    pooled_auc = roc_auc_score(y, pooled_pred)
    pooled_slope, pooled_intercept = calibration_slope_intercept(y, pooled_pred)
    fpr, tpr, _ = roc_curve(y, pooled_pred)

    print(f"\n{'=' * 60}\nPooled logistic regression results "
          f"({args.n_imputations} imputations, 5-fold CV each)\n{'=' * 60}")
    print(f"Pooled CV-AUC: {pooled_auc:.3f}")
    print(f"Pooled calibration slope={pooled_slope:.3f}, intercept={pooled_intercept:.3f}")

    print("\nTop 15 coefficients by |magnitude| (imputation 1, standardized features):")
    print(all_coefs.reindex(all_coefs.abs().sort_values(ascending=False).index).head(15))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Logistic regression, 5-fold CV (AUC={pooled_auc:.3f})",
            color="tab:green")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(
        "Logistic regression (Model 3 features) - ROC curve\n"
        f"Pooled 5-fold CV AUC={pooled_auc:.3f} ({args.n_imputations} imputations)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_out_path = REPO_ROOT / "logreg_comparison_roc_curve.png"
    fig.savefig(roc_out_path, dpi=150)
    print(f"\nROC curve saved to: {roc_out_path}")

    obs_freq, pred_freq = calibration_curve(y, pooled_pred, n_bins=10, strategy="quantile")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(pred_freq, obs_freq, marker="o",
              label=f"Logistic regression (slope={pooled_slope:.3f}, intercept={pooled_intercept:.3f})",
              color="tab:green")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "Logistic regression (Model 3 features) - Calibration plot\n"
        f"Pooled slope={pooled_slope:.3f}, intercept={pooled_intercept:.3f}"
    )
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    cal_out_path = REPO_ROOT / "logreg_comparison_calibration_curve.png"
    fig2.savefig(cal_out_path, dpi=150)
    print(f"Calibration curve saved to: {cal_out_path}")


if __name__ == "__main__":
    main()
