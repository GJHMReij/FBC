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

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pe_model1_cbc_rf import (
    AGE_COL, CREATININE_COL, DEFAULT_INPUT_CSV, OUTCOME_COL, PROTECTED_COLS,
    REPO_ROOT, SENSITIVITY_TARGETS, SEX_COL,
    bootstrap_optimism, calculate_correlation, calibration_slope_intercept,
    clean_data, fit_pipeline, pearson_filter, read_dictionary,
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
    X0 = imputed_datasets[0]
    selected0 = calculate_correlation(X0, model3_features, threshold=0.9, verbose=False)
    selected0 = pearson_filter(X0, y, selected0, p_threshold=0.8, verbose=False)
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
    grid_search.fit(X0_scaled, y)
    rf_params = grid_search.best_params_
    print(f"Best params (fixed for all imputations): {rf_params}")
    print(f"Best CV AUC (imputation 1 only): {grid_search.best_score_:.3f}")

    per_imputation_results = []
    for m, X_m in enumerate(imputed_datasets):
        print(f"\n--- Imputation {m + 1}/{args.n_imputations} ---")
        model, scaler, selected = fit_pipeline(X_m, y, model3_features, rf_params)
        apparent_pred = model.predict_proba(scaler.transform(X_m[selected]))[:, 1]
        apparent_auc = roc_auc_score(y, apparent_pred)
        apparent_slope, apparent_intercept = calibration_slope_intercept(y, apparent_pred)
        apparent_brier = brier_score_loss(y, apparent_pred)

        optimism = bootstrap_optimism(
            X_m, y, model3_features, rf_params, n_boot=args.n_bootstrap, random_state=200 + m
        )

        corrected_auc = apparent_auc - optimism["auc"].mean()
        corrected_slope = apparent_slope - optimism["slope"].mean()
        corrected_intercept = apparent_intercept - optimism["intercept"].mean()
        corrected_brier = apparent_brier - optimism["brier"].mean()

        threshold_corrected = {}
        for target in SENSITIVITY_TARGETS:
            apparent_metrics = sensitivity_threshold_metrics(y, apparent_pred, target)
            threshold_corrected[target] = {
                key: apparent_metrics[key] - optimism[f"{target}_{key}"].mean()
                for key in ("sensitivity", "specificity", "ppv", "npv", "efficiency")
            }

        print(f"  Corrected AUC={corrected_auc:.3f}, Brier={corrected_brier:.4f}, "
              f"slope={corrected_slope:.3f}, intercept={corrected_intercept:.3f}")

        importances = pd.Series(model.feature_importances_, index=selected)
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
        for key in ("sensitivity", "specificity", "ppv", "npv", "efficiency"):
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

    fig, ax = plt.subplots(figsize=(6, 6))
    for m, X_m in enumerate(imputed_datasets):
        model, scaler, selected = fit_pipeline(X_m, y, model3_features, rf_params)
        pred_m = model.predict_proba(scaler.transform(X_m[selected]))[:, 1]
        fpr, tpr, _ = roc_curve(y, pred_m)
        ax.plot(fpr, tpr, color="tab:blue", alpha=0.3,
                 label="Per-imputation (apparent)" if m == 0 else None)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(
        "Model 3 (Model 2 + D-dimer level + assay) - ROC curves\n"
        f"Pooled bootstrap-corrected AUC={pooled['auc'][0]:.3f} "
        f"(SE={pooled['auc'][1]:.3f}, {args.n_imputations} imputations x "
        f"{args.n_bootstrap} resamples)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_out_path = REPO_ROOT / "model3_cbc_diff_ddimer_roc_curve.png"
    fig.savefig(roc_out_path, dpi=150)
    print(f"\nROC curve saved to: {roc_out_path}")


if __name__ == "__main__":
    main()
