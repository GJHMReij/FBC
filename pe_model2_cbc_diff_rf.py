"""
PE prediction - Model 2: age + sex + creatinine + all CBC + DIFF parameters.

Implements Model 2 from "Analysis plan FBC versie 17-12": Model 1 plus the
white blood cell differential (DIFF) panel (~159 parameters total, per
Table 1). Patients whose Sysmex order was CBC-only have no DIFF values; per
the analysis plan those are imputed via multiple imputation by chained
equations (MICE, 10 imputation sets), with results pooled across imputations
using Rubin's rule. Shares all data-cleaning / feature-selection / bootstrap
machinery with pe_model1_cbc_rf.py (imported directly, not duplicated).

Run with the project venv:
    .venv/bin/python pe_model2_cbc_diff_rf.py

By default this reads the synthetic dummy dataset next to this script. On
MyDRE, point it at the real cohort CSV instead, without touching the code:
    .venv/bin/python pe_model2_cbc_diff_rf.py --input-csv /path/to/real_cohort.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

from pe_model1_cbc_rf import (
    AGE_COL, CREATININE_COL, DEFAULT_INPUT_CSV, OUTCOME_COL, REPO_ROOT,
    SENSITIVITY_TARGETS, SEX_COL,
    build_discrete_channel_map, build_feature_channel_map,
    bootstrap_optimism, calculate_correlation, calibration_slope_intercept,
    clean_data, EXTRA_CHANNELS, CBC_CHANNELS, fit_pipeline,
    metrics_at_threshold, pearson_filter, read_dictionary,
    sensitivity_threshold_metrics,
)

N_IMPUTATIONS = 10  # per analysis plan

# DIFF is added on top of Model 1's CBC channels; RET/PLT-F/WPC stay
# out-of-scope for Model 2 too (Table 1 is explicitly "159 parameters
# (CBC+DIFF)", not the extended panels).
CBC_DIFF_CHANNELS = CBC_CHANNELS | {"DIFF/WDF"}

# Auxiliary variables the analysis plan asks the imputation model to
# include, beyond the DIFF parameters themselves and the Model 1 covariates
# (age/sex/creatinine, already in the feature matrix). Quantitative D-dimer
# level isn't in the dummy dataset (Model 3 concern) and calendar year needs
# a real, parseable order date -- both are picked up automatically here if
# and when they're present and usable, so this degrades gracefully on the
# dummy data and "just works" on the real MyDRE cohort.
HOSPITAL_COL = "Ziekenhuislocatie"
ORDER_DATE_COL = "OnderzoeksDatum"


def get_cbc_diff_features(feature_channel_map):
    features = []
    for feat, chmap in feature_channel_map.items():
        all_ch = set(chmap["YES"]) | set(chmap["OPTION"])
        if not all_ch:
            continue
        if EXTRA_CHANNELS & all_ch:
            continue  # needs RET / PLT-F / WPC -> out of scope for Model 2
        if all_ch.issubset(CBC_DIFF_CHANNELS):
            features.append(feat)
    return features


def build_imputation_frame(df, y, model2_features):
    """Assemble the matrix MICE imputes over: the Model 2 feature columns
    (age/sex/creatinine/CBC/DIFF, some with missing DIFF values) plus
    auxiliary variables that help predict the missingness but are not
    themselves Model 2 predictors (hospital, outcome), added per the
    analysis plan's imputation-model requirement. Auxiliary columns that
    aren't usable in this particular input file are skipped with a warning
    rather than failing, so the same code runs on both the dummy and the
    real MyDRE cohort.
    """
    aux = pd.DataFrame(index=df.index)

    if HOSPITAL_COL in df.columns and df[HOSPITAL_COL].notna().any():
        aux = pd.concat([aux, pd.get_dummies(df[HOSPITAL_COL], prefix="hosp", dtype=float)], axis=1)
        print(f"Imputation model: including hospital ({HOSPITAL_COL})")
    else:
        print(f"Imputation model: hospital column '{HOSPITAL_COL}' not usable, skipping")

    if ORDER_DATE_COL in df.columns:
        order_year = pd.to_datetime(df[ORDER_DATE_COL], errors="coerce").dt.year
        if order_year.notna().any():
            aux["calendar_year"] = order_year.astype(float)
            print(f"Imputation model: including calendar year (from {ORDER_DATE_COL})")
        else:
            print(f"Imputation model: '{ORDER_DATE_COL}' not parseable as a date in this "
                  f"file (e.g. scrubbed dummy placeholders), skipping calendar year")
    else:
        print(f"Imputation model: date column '{ORDER_DATE_COL}' not found, skipping calendar year")

    aux["outcome"] = y.astype(float)

    imputation_frame = pd.concat([df[model2_features], aux], axis=1)
    return imputation_frame


def run_mice(imputation_frame, model2_features, n_imputations, random_state=42):
    """Run IterativeImputer n_imputations times with sample_posterior=True
    and different random states, producing n_imputations independent
    multiply-imputed versions of the Model 2 feature columns (proper MI,
    not a single deterministic fill)."""
    imputed_datasets = []
    for m in range(n_imputations):
        imputer = IterativeImputer(
            sample_posterior=True, random_state=random_state + m,
            max_iter=10, initial_strategy="median",
        )
        imputed = imputer.fit_transform(imputation_frame)
        imputed_df = pd.DataFrame(imputed, columns=imputation_frame.columns, index=imputation_frame.index)
        imputed_datasets.append(imputed_df[model2_features])
        print(f"  MICE imputation {m + 1}/{n_imputations} done")
    return imputed_datasets


def rubin_pool(estimates, within_variances):
    """Pool point estimates and their within-imputation variances across
    multiple-imputation datasets using Rubin's rule.
    Returns (pooled_estimate, pooled_variance, pooled_se)."""
    m = len(estimates)
    pooled_estimate = np.mean(estimates)
    within_var = np.mean(within_variances)
    between_var = np.var(estimates, ddof=1) if m > 1 else 0.0
    total_var = within_var + between_var + between_var / m
    return pooled_estimate, total_var, np.sqrt(total_var)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="Path to the cohort CSV (defaults to the synthetic dummy dataset)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=100,
        help="Bootstrap resamples per imputation (default: 100 -- lower than Model "
             "1's 500 since this runs once per MICE imputation; total cost is "
             "n_imputations x n_bootstrap model fits)",
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
    y = (df[OUTCOME_COL] == "Ja").astype(int)

    model2_features = ["Geslacht_enc", AGE_COL, CREATININE_COL] + numeric_features
    print(f"\nModel 2 feature count: {len(model2_features)}")
    print(f"Outcome distribution:\n{y.value_counts()}")

    # Drop rows missing the Model 1 covariates or with no usable outcome --
    # these can't be fixed by MICE (which imputes the DIFF panel, not core
    # covariates/outcome). DIFF-only missingness is left in place for MICE.
    core_cols = ["Geslacht_enc", AGE_COL, CREATININE_COL]
    complete_core_mask = df[core_cols].notna().all(axis=1) & y.notna()
    n_dropped = (~complete_core_mask).sum()
    print(f"\nDropping {n_dropped} rows with missing core covariates (age/sex/creatinine) "
          f"or outcome ({n_dropped / len(df) * 100:.1f}%) -- MICE covers DIFF-panel "
          f"missingness only, not these")
    df = df.loc[complete_core_mask].reset_index(drop=True)
    y = y.loc[complete_core_mask].reset_index(drop=True)

    diff_missing_frac = df[numeric_features].isnull().mean().mean()
    print(f"Mean missingness across DIFF-eligible features: {diff_missing_frac * 100:.1f}%")

    imputation_frame = build_imputation_frame(df, y, model2_features)

    print(f"\nRunning MICE ({args.n_imputations} imputations)...")
    imputed_datasets = run_mice(imputation_frame, model2_features, args.n_imputations)

    # Tune hyperparameters ONCE, on the first imputed dataset, then reuse
    # across all imputations and their bootstrap loops. Missing-DIFF
    # imputation doesn't change the overfitting/complexity tradeoff the grid
    # search is tuned for, and re-tuning per imputation would multiply the
    # already-expensive grid search by n_imputations for no real benefit.
    print("\nTuning hyperparameters once, on the first imputed dataset...")
    X0 = imputed_datasets[0]
    selected0 = calculate_correlation(X0, model2_features, threshold=0.9, verbose=False)
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

    # Per imputation: fit the apparent model, then run bootstrap optimism
    # correction (same machinery as Model 1). Collect the corrected point
    # estimate AND its within-imputation bootstrap variance for Rubin pooling.
    per_imputation_results = []
    for m, X_m in enumerate(imputed_datasets):
        print(f"\n--- Imputation {m + 1}/{args.n_imputations} ---")
        model, scaler, selected = fit_pipeline(X_m, y, model2_features, rf_params)
        apparent_pred = model.predict_proba(scaler.transform(X_m[selected]))[:, 1]
        apparent_auc = roc_auc_score(y, apparent_pred)
        apparent_slope, apparent_intercept = calibration_slope_intercept(y, apparent_pred)
        apparent_brier = brier_score_loss(y, apparent_pred)

        optimism = bootstrap_optimism(
            X_m, y, model2_features, rf_params, n_boot=args.n_bootstrap, random_state=100 + m
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

        per_imputation_results.append({
            "auc": corrected_auc, "auc_var": optimism["auc"].var(ddof=1),
            "slope": corrected_slope, "slope_var": optimism["slope"].var(ddof=1),
            "intercept": corrected_intercept, "intercept_var": optimism["intercept"].var(ddof=1),
            "brier": corrected_brier, "brier_var": optimism["brier"].var(ddof=1),
            "threshold": threshold_corrected,
            "optimism": optimism,
        })

    # Pool across imputations via Rubin's rule.
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
            # Within-imputation variance for threshold metrics: reuse the
            # per-resample optimism spread as a proxy (same approach as
            # auc/slope/etc above, applied per threshold/metric).
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

    # Plots: overlay each imputation's apparent ROC/calibration curve (thin,
    # transparent) to visualise between-imputation spread, with the pooled
    # AUC/calibration numbers in the title.
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, X_m in enumerate(imputed_datasets):
        model, scaler, selected = fit_pipeline(X_m, y, model2_features, rf_params)
        pred_m = model.predict_proba(scaler.transform(X_m[selected]))[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, pred_m)
        ax.plot(fpr, tpr, color="tab:blue", alpha=0.3,
                 label="Per-imputation (apparent)" if m == 0 else None)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(
        "Model 2 (age + sex + creatinine + CBC + DIFF) - ROC curves\n"
        f"Pooled bootstrap-corrected AUC={pooled['auc'][0]:.3f} "
        f"(SE={pooled['auc'][1]:.3f}, {args.n_imputations} imputations x "
        f"{args.n_bootstrap} resamples)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_out_path = REPO_ROOT / "model2_cbc_diff_roc_curve.png"
    fig.savefig(roc_out_path, dpi=150)
    print(f"\nROC curve saved to: {roc_out_path}")


if __name__ == "__main__":
    main()
