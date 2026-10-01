"""
PE prediction - Model 3: Model 2 + quantitative D-dimer level + D-dimer assay.

Implements Model 3 from "Analysis plan FBC versie 17-12": Model 2 (age + sex +
creatinine + CBC + DIFF) plus quantitative D-dimer level and D-dimer assay
type. The assay column already distinguishes VUmc's pre-2020-03-03 Tinaquant
(Roche) era from the Innovance (Siemens) assay used since -- exactly the
switch the analysis plan says "will be taken into consideration during the
analysis" -- so including it as a covariate (rather than needing a separate
order-date cutoff) directly captures that. D-dimer level and assay are
treated as required Model 3 covariates (like age/sex/creatinine in Model 1):
rows missing either are dropped, they are not MICE-imputed themselves, but
per the analysis plan they ARE included as predictors in the MICE model that
imputes the missing DIFF panel. Shares all machinery with pe_model1_cbc_rf.py
and pe_model2_cbc_diff_rf.py (imported directly, not duplicated).

Run with the project venv:
    .venv/bin/python pe_model3_cbc_diff_ddimer_rf.py

By default this reads the synthetic dummy dataset next to this script. On
MyDRE, point it at the real cohort CSV instead, without touching the code:
    .venv/bin/python pe_model3_cbc_diff_ddimer_rf.py --input-csv /path/to/real_cohort.csv
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pe_model1_cbc_rf import (
    AGE_COL, CREATININE_COL, DEFAULT_INPUT_CSV, Heartbeat, NOT_ASSESSABLE_VALUE,
    ORDER_ID_COL, OUTCOME_COL, OUTCOME_CORRECTION_COL, PROTECTED_COLS,
    REPO_ROOT, SENSITIVITY_TARGETS, SEX_COL,
    apply_outcome_correction, bootstrap_optimism, calculate_correlation,
    calibration_slope_intercept, clean_data, fit_pipeline, pearson_filter,
    read_dictionary, save_manuscript_data, save_roc_data,
    sensitivity_threshold_metrics, build_feature_channel_map,
)
from pe_model2_cbc_diff_rf import (
    D_DIMER_ASSAY_COL, D_DIMER_ASSAY_MAP, D_DIMER_VALUE_COL, N_IMPUTATIONS,
    build_imputation_frame, get_cbc_diff_features, run_mice, rubin_pool,
)
# D_DIMER_ASSAY_MAP: per proposal, "VU switched to the Innovance assay from
# Siemens ... already in use at AMC" on 2020-03-03. Innovance=1 as the
# encoding reference; the assay column itself already separates the two
# eras, no date needed.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="Path to the cohort CSV (defaults to the synthetic dummy dataset)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=100,
        help="Bootstrap resamples per imputation (default: 100; use 500 on "
             "the real MyDRE cohort to match the analysis plan)",
    )
    parser.add_argument(
        "--n-imputations", type=int, default=N_IMPUTATIONS,
        help=f"Number of MICE imputation sets (default: {N_IMPUTATIONS}, per analysis plan)",
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
    # Defensive: strip stray whitespace/control characters from column names
    # (e.g. a trailing \r on the last column of a Windows-exported CSV) that
    # would otherwise silently break exact-name lookups like df["Geslacht"].
    df.columns = df.columns.str.strip()
    print(f"Shape: {df.shape}")

    if args.outcome_correction is not None:
        print(f"\nApplying outcome correction from {args.outcome_correction} ...")
        df = apply_outcome_correction(df, args.outcome_correction)

    dictionary_df = read_dictionary()
    feature_channel_map = build_feature_channel_map(dictionary_df)
    cbc_diff_features = get_cbc_diff_features(feature_channel_map)
    cbc_diff_features = [c for c in cbc_diff_features if c in df.columns]
    print(f"N candidate CBC+DIFF features from dictionary: {len(cbc_diff_features)}")

    df, cbc_diff_features = clean_data(df, cbc_diff_features)

    numeric_features = [
        c for c in cbc_diff_features
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    print(f"N numeric CBC+DIFF features after cleaning: {len(numeric_features)}")

    sex_map = {"Man": 1.0, "Vrouw": 0.0}
    df["Geslacht_enc"] = df[SEX_COL].map(sex_map)
    df["D_dimer_assay_enc"] = df[D_DIMER_ASSAY_COL].map(D_DIMER_ASSAY_MAP)
    y = (df[OUTCOME_COL] == "Ja").astype(int)

    model3_features = (
        ["Geslacht_enc", AGE_COL, CREATININE_COL, D_DIMER_VALUE_COL, "D_dimer_assay_enc"]
        + numeric_features
    )
    print(f"\nModel 3 feature count: {len(model3_features)}")
    print(f"Outcome distribution:\n{y.value_counts()}")

    # D-dimer level/assay are required Model 3 covariates -- like
    # age/sex/creatinine in Model 1, they're dropped-if-missing rather than
    # MICE-imputed (MICE here is reserved for the DIFF panel, per the
    # analysis plan). D-dimer assay is required so this run's calibration
    # correctly reflects the VUmc 2020-03-03 assay-switch covariate.
    PROTECTED_COLS.update({D_DIMER_VALUE_COL, "D_dimer_assay_enc"})
    core_cols = ["Geslacht_enc", AGE_COL, CREATININE_COL, D_DIMER_VALUE_COL, "D_dimer_assay_enc"]
    complete_core_mask = df[core_cols].notna().all(axis=1) & y.notna()
    n_dropped = (~complete_core_mask).sum()
    print(f"\nDropping {n_dropped} rows with missing core covariates "
          f"(age/sex/creatinine/D-dimer level/D-dimer assay) or outcome "
          f"({n_dropped / len(df) * 100:.1f}%) -- MICE covers DIFF-panel "
          f"missingness only, not these")
    df = df.loc[complete_core_mask].reset_index(drop=True)
    y = y.loc[complete_core_mask].reset_index(drop=True)

    print(f"N remaining after core-covariate drop: {len(df)} (events={y.sum()})")

    diff_missing_frac = df[numeric_features].isnull().mean().mean()
    print(f"Mean missingness across DIFF-eligible features: {diff_missing_frac * 100:.1f}%")

    # D-dimer level/assay are included as auxiliary predictors in the MICE
    # imputation model for the DIFF panel (per the analysis plan), in
    # addition to being final Model 3 covariates -- build_imputation_frame
    # picks them up automatically since they're already in model3_features.
    imputation_frame = build_imputation_frame(df, y, model3_features)

    print(f"\nRunning MICE ({args.n_imputations} imputations)...")
    imputed_datasets = run_mice(imputation_frame, model3_features, args.n_imputations)

    print("\nTuning hyperparameters once, on the first imputed dataset...")
    print("Skipping correlation + Pearson-association filters (uniform with Model 1 as of "
          "2026-10-01): using all candidate features")
    X0 = imputed_datasets[0]
    selected0 = list(model3_features)
    scaler0 = StandardScaler()
    X0_scaled = scaler0.fit_transform(X0[selected0])
    base_model = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced", oob_score=False
    )
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "min_samples_leaf": [10, 25, 50],
        "min_samples_split": [20, 50, 100],
        "max_features": ["sqrt", 0.2, 0.3],
    }
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0,
    )
    with Heartbeat("GridSearchCV"):
        grid_search.fit(X0_scaled, y)
    rf_params = grid_search.best_params_
    print(f"Best params (fixed for all imputations): {rf_params}")
    print(f"Best CV AUC (imputation 1 only): {grid_search.best_score_:.3f}")

    per_imputation_results = []
    imputation1_importances = None
    all_oob_preds = []
    fitted_imputation_models = []
    for m, X_m in enumerate(imputed_datasets):
        print(f"\n--- Imputation {m + 1}/{args.n_imputations} ---")
        model, scaler, selected = fit_pipeline(X_m, y, model3_features, rf_params, skip_correlation_filter=True, skip_pearson_filter=True)
        fitted_imputation_models.append({"model": model, "scaler": scaler, "features": selected})
        apparent_pred = model.predict_proba(scaler.transform(X_m[selected]))[:, 1]
        apparent_auc = roc_auc_score(y, apparent_pred)
        apparent_slope, apparent_intercept = calibration_slope_intercept(y, apparent_pred)
        apparent_brier = brier_score_loss(y, apparent_pred)

        optimism, oob_pred_avg = bootstrap_optimism(
            X_m, y, model3_features, rf_params, n_boot=args.n_bootstrap, random_state=200 + m,
            skip_correlation_filter=True, skip_pearson_filter=True,
        )
        all_oob_preds.append(oob_pred_avg)

        corrected_auc = apparent_auc - optimism["auc"].mean()
        corrected_slope = apparent_slope - optimism["slope"].mean()
        corrected_intercept = apparent_intercept - optimism["intercept"].mean()
        corrected_brier = apparent_brier - optimism["brier"].mean()

        threshold_corrected = {}
        for target in SENSITIVITY_TARGETS:
            apparent_metrics = sensitivity_threshold_metrics(y, apparent_pred, target)
            threshold_corrected[target] = {
                key: apparent_metrics[key] - optimism[f"{target}_{key}"].mean()
                for key in ("sensitivity", "specificity", "ppv", "npv", "efficiency", "accuracy", "f1")
            }

        print(f"  Corrected AUC={corrected_auc:.3f}, Brier={corrected_brier:.4f}, "
              f"slope={corrected_slope:.3f}, intercept={corrected_intercept:.3f}")

        importances = pd.Series(model.feature_importances_, index=selected)
        if m == 0:
            # Feature importance ranking (for the manuscript's Table 3) is
            # taken from imputation 1 only, consistent with hyperparameter
            # tuning also using only imputation 1.
            imputation1_importances = importances
        d_dimer_rank = importances.rank(ascending=False)
        print(f"  D-dimer level importance rank: {int(d_dimer_rank.get(D_DIMER_VALUE_COL, -1))} "
              f"of {len(selected)} (value={importances.get(D_DIMER_VALUE_COL, float('nan')):.4f})")

        per_imputation_results.append({
            "auc": corrected_auc, "auc_var": optimism["auc"].var(ddof=1),
            "slope": corrected_slope, "slope_var": optimism["slope"].var(ddof=1),
            "intercept": corrected_intercept, "intercept_var": optimism["intercept"].var(ddof=1),
            "brier": corrected_brier, "brier_var": optimism["brier"].var(ddof=1),
            "threshold": threshold_corrected,
            "optimism": optimism,
        })

    # Save all 10 fitted imputation models (+ their scalers/feature lists)
    # and the fixed rf_params, so they can later be applied unchanged to
    # external data (e.g. UCLH/Barts) without retraining. How to combine
    # the 10 models' predictions for a new patient is a separate, not-yet-
    # built step; this just persists what would be needed for that later.
    model_out_path = REPO_ROOT / "model3_fitted.joblib"
    joblib.dump(
        {"imputation_models": fitted_imputation_models, "rf_params": rf_params},
        model_out_path,
    )
    print(f"\nFitted models (all {args.n_imputations} imputations) saved to: {model_out_path}")

    print(f"\n{'=' * 60}\nPooled results across {args.n_imputations} MICE imputations "
          f"(Rubin's rule)\n{'=' * 60}")

    pooled = {}
    for key in ("auc", "slope", "intercept", "brier"):
        estimates = [r[key] for r in per_imputation_results]
        variances = [r[f"{key}_var"] for r in per_imputation_results]
        pooled_est, pooled_var, pooled_se = rubin_pool(estimates, variances)
        pooled[key] = (pooled_est, pooled_se)
        print(f"Pooled {key}: {pooled_est:.4f} (SE={pooled_se:.4f}, "
              f"95% CI=[{pooled_est - 1.96 * pooled_se:.4f}, {pooled_est + 1.96 * pooled_se:.4f}])")

    pooled_threshold_rows = []
    for target in SENSITIVITY_TARGETS:
        row = {"target_sensitivity": target}
        for key in ("sensitivity", "specificity", "ppv", "npv", "efficiency", "accuracy", "f1"):
            estimates = [r["threshold"][target][key] for r in per_imputation_results]
            variances = [
                r["optimism"][f"{target}_{key}"].var(ddof=1) for r in per_imputation_results
            ]
            pooled_est, _, pooled_se = rubin_pool(estimates, variances)
            row[key] = pooled_est
            row[f"{key}_se"] = pooled_se
        pooled_threshold_rows.append(row)
    pooled_threshold_df = pd.DataFrame(pooled_threshold_rows).set_index("target_sensitivity")

    print(f"\nPooled performance at fixed sensitivity thresholds "
          f"(bootstrap-corrected + Rubin-pooled across imputations):")
    print(pooled_threshold_df.round(3))

    # Pool OOB (out-of-bag) predictions across the 10 imputations: for each
    # patient, average their OOB prediction (already itself an average
    # across ~184 bootstrap resamples, see bootstrap_optimism()) over
    # whichever imputations gave a valid value (nanmean -- a patient could
    # in rare cases be never-OOB within one imputation's 500 resamples).
    # This single pooled array is a genuine internal-validation prediction
    # per patient (never trained on by the models that produced it), unlike
    # the apparent per-imputation curves previously plotted here, which
    # didn't match the pooled bootstrap-corrected numbers shown alongside
    # them (same mismatch issue fixed for Model 1).
    with np.errstate(invalid="ignore"):
        oob_pred_pooled = np.nanmean(np.array(all_oob_preds), axis=0)
    oob_mask = ~np.isnan(oob_pred_pooled)
    y_oob = y.to_numpy()[oob_mask]
    oob_pred_valid = oob_pred_pooled[oob_mask]
    oob_auc = roc_auc_score(y_oob, oob_pred_valid)
    oob_fpr, oob_tpr, _ = roc_curve(y_oob, oob_pred_valid)
    print(f"\nPooled internal validation (OOB) AUC: {oob_auc:.3f} "
          f"(n={oob_mask.sum()} patients with >=1 out-of-bag prediction)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(oob_fpr, oob_tpr,
            label=f"Internal validation, OOB (AUC={oob_auc:.3f})", color="tab:orange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(
        "Model 3 (Model 2 + D-dimer level + assay) - ROC curve\n"
        f"Pooled bootstrap-corrected AUC={pooled['auc'][0]:.3f} "
        f"(SE={pooled['auc'][1]:.3f}, {args.n_imputations} imputations x "
        f"{args.n_bootstrap} resamples)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_out_path = REPO_ROOT / "model3_cbc_diff_ddimer_roc_curve.png"
    fig.savefig(roc_out_path, dpi=150, bbox_inches="tight")
    print(f"\nROC curve saved to: {roc_out_path}")

    save_roc_data("Model 3 (+D-dimer)", oob_fpr, oob_tpr, pooled["auc"][0],
                  REPO_ROOT / "model3_roc_data.json")

    # Calibration plot (new -- Model 3 previously had no calibration plot at
    # all). Uses the same pooled OOB predictions as the ROC curve above.
    oob_obs_freq, oob_pred_freq = calibration_curve(y_oob, oob_pred_valid, n_bins=10, strategy="quantile")
    oob_cal_slope, oob_cal_intercept = calibration_slope_intercept(y_oob, oob_pred_valid)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(oob_pred_freq, oob_obs_freq, marker="o",
              label=f"Internal validation, OOB (slope={oob_cal_slope:.3f}, intercept={oob_cal_intercept:.3f})",
              color="tab:orange")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "Model 3 (Model 2 + D-dimer level + assay) - Calibration plot\n"
        f"Pooled bootstrap-corrected slope={pooled['slope'][0]:.3f}, "
        f"intercept={pooled['intercept'][0]:.3f} ({args.n_imputations} imputations x "
        f"{args.n_bootstrap} resamples)"
    )
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    cal_out_path = REPO_ROOT / "model3_cbc_diff_ddimer_calibration_curve.png"
    fig2.savefig(cal_out_path, dpi=150, bbox_inches="tight")
    print(f"Calibration curve saved to: {cal_out_path}")

    save_manuscript_data(
        "Model 3 (+D-dimer)", pooled["auc"][0], pooled["brier"][0],
        pooled_threshold_df.loc[0.95].to_dict(),
        list(imputation1_importances.sort_values(ascending=False).head(10).items()),
        REPO_ROOT / "model3_manuscript_data.json",
    )


if __name__ == "__main__":
    main()
