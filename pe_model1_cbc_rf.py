"""
PE prediction - Model 1: age + sex + creatinine + all CBC parameters.

Implements Model 1 from "Analysis plan FBC versie 17-12": a Random Forest
classifier predicting pulmonary embolism (PE) from demographics + CBC-only
Sysmex parameters.

Data cleaning follows the approach in
full_data_processing/fbc_helper_scripts/zero_na_pipelineV001_KD_11_08_2025.py
and get_exclude_features.py: the Sysmex analyser fills unmeasured parameters
with 0 rather than NaN, so we use the Sysmex data dictionary + the sample's
`Discrete` (channel combination) value to convert those false zeros to NaN,
and drop variables flagged for exclusion.

Run with the project venv:
    .venv/bin/python pe_model1_cbc_rf.py

By default this reads the synthetic dummy dataset next to this script. On
MyDRE, point it at the real cohort CSV instead, without touching the code:
    .venv/bin/python pe_model1_cbc_rf.py --input-csv /path/to/real_cohort.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

from scipy.special import logit
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Columns that should never be dropped by the correlation/association
# filters below, even if they correlate with or weakly associate with other
# CBC features -- they are the required covariates from the analysis plan.
PROTECTED_COLS = {"Geslacht_enc", "leeftijd", "creatinine"}

# Standalone folder: dictionary, dummy CSV, and outputs all live next to this
# script, not inside the git repo.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR  # ROC curve output path below is relative to this
DICTIONARY_PATH = SCRIPT_DIR / "SysmexDictionary_2025_10_08_KD.ods"
DICTIONARY_SHEET = "500set"

# Default: synthetic dummy dataset for local development (no real patient
# data should live on a personal laptop). On MyDRE, pass --input-csv to point
# at the real cohort CSV instead -- no code change needed.
DEFAULT_INPUT_CSV = SCRIPT_DIR / "CohortML_DUMMY.csv"

OUTCOME_COL = "PE_according_to_all_sources"
AGE_COL = "leeftijd"
SEX_COL = "Geslacht"
CREATININE_COL = "creatinine"
DISCRETE_COL = "Discrete"

PLACEHOLDER_STRINGS = ["----"]

# From get_exclude_features.py (KD/Didier's manual review)
EXCLUDED_VARS = [
    'WBC_BF', 'WBC_BF_MARK', 'RBC_BF', 'RBC_BF_MARK', 'TC_BF_COUNT', 'TC_BF_COUNT_MARK',
    'HF_BF_COUNT', 'HF_BF_COUNT_MARK', 'HF_BF_PERC_100WBC', 'HF_BF_PERC_MARK',
    'NE_BF_COUNT', 'NE_BF_COUNT_MARK', 'NE_BF_PERC_PERC', 'NE_BF_PERC_MARK',
    'LY_BF_COUNT', 'LY_BF_COUNT_MARK', 'LY_BF_PERC_PERC', 'LY_BF_PERC_MARK',
    'MO_BF_COUNT', 'MO_BF_COUNT_MARK', 'MO_BF_PERC_PERC', 'MO_BF_PERC_MARK',
    'EO_BF_COUNT', 'EO_BF_COUNT_MARK', 'EO_BF_PERC_PERC', 'EO_BF_PERC_MARK',
    'RBC_BF2', 'RBC_BF2_MARK',
]
HS_VARS = [
    'WBC_hsA', 'WBC_hsA_MARK', 'RBC_hsA', 'RBC_hsA_MARK', 'RBC_I_hsA', 'RBC_I_hsA_MARK',
    'RBC_O_hsA', 'RBC_O_hsA_MARK', 'NEUT_COUNT_hsA', 'NEUT_COUNT_hsA_MARK',
    'LYMPHO_COUNT_hsA', 'LYMPHO_COUNT_hsA_MARK', 'MONO_COUNT_hsA', 'MONO_COUNT_hsA_MARK',
    'EO_COUNT_hsA', 'EO_COUNT_hsA_MARK', 'NEUT_PERC_hsA_PERC', 'NEUT_PERC_hsA_MARK',
    'LYMPHO_PERC_hsA_PERC', 'LYMPHO_PERC_hsA_MARK', 'MONO_PERC_hsA_PERC', 'MONO_PERC_hsA_MARK',
    'EO_PERC_hsA_PERC', 'EO_PERC_hsA_MARK', 'MN_COUNT_hsA', 'MN_COUNT_hsA_MARK',
    'PMN_COUNT_hsA', 'PMN_COUNT_hsA_MARK', 'HF_COUNT_hsA', 'HF_COUNT_hsA_MARK',
    'MN_PERC_hsA_PERC', 'MN_PERC_hsA_MARK', 'PMN_PERC_hsA_PERC', 'PMN_PERC_hsA_MARK',
    'HF_PERC_hsA_100WBC', 'HF_PERC_hsA_MARK', 'TC_COUNT_hsA', 'TC_COUNT_hsA_MARK',
]
NAN_ALWAYS_VARS = ['IP_SUS_RBC_pRBC']

# From helper_process_methods.py (Kasia): IP flags where missing == "flag not
# triggered" rather than genuinely unknown -> fillna(0), not imputed.
IP_FILL_NA = [
    "IP_ABN_WBC_NRBC_Present",
    "IP_ABN_RBC_RBC_Abn_Distribution",
    "IP_ABN_RBC_Dimorphic_Population",
    "IP_ABN_RBC_Anisocytosis",
    "IP_ABN_RBC_Microcytosis",
    "IP_ABN_RBC_Macrocytosis",
    "IP_ABN_RBC_Hypochromia",
    "IP_ABN_RBC_Anemia",
    "IP_ABN_RBC_Erythrocytosis",
    "IP_ABN_PLT_PLT_Abn_Distribution",
    "IP_ABN_PLT_Thrombocytopenia",
    "IP_ABN_PLT_Thrombocytosis",
    "IP_SUS_WBC_Blasts",
    "IP_SUS_RBC_RBC_Agglutination",
    "IP_SUS_RBC_Turbidity_HGB_Interf",
    "IP_SUS_RBC_Iron_Deficiency",
    "IP_SUS_RBC_HGB_Defect",
    "IP_SUS_RBC_Fragments",
    "IP_SUS_PLT_PLT_Clumps",
]
# From get_exclude_features.py (Kasia/Didier): treat as categorical rather
# than numeric.
CATEGORICAL_VARS = [
    "Q_Flag_Turbidity_HGB_Interf", "Q_Flag_Iron_Deficiency",
    "Q_Flag_HGB_Defect", "Q_Flag_Fragments",
]
# Restricted features: always set to NaN regardless of Discrete/channel (per
# email communication 1-12-2025, mirrored from zero_na_pipeline).
RESTRICTED_FEATURES = [
    "NEUT_COUNT",
    "NEUT_PERC_PERC",
    "NEUT_COUNT_minus_IG_COUNT",
    "NEUT_PERC_minus_IG_PERC",
]

CBC_CHANNELS = {"CBC-RBC/PLT", "CBC-HGB", "CBC-WNR"}
EXTRA_CHANNELS = {"RET", "PLT-F", "WPC"}  # channels beyond basic CBC+DIFF

APPLICATION_CHANNELS = {
    "CBC": ["CBC-RBC/PLT", "CBC-HGB", "CBC-WNR"],
    "DIFF": ["CBC-WNR", "DIFF/WDF"],
    "RET": ["CBC-RBC/PLT", "RET"],
    "PLT-F": ["CBC-RBC/PLT", "PLT-F"],
    "WPC": ["CBC-WNR", "DIFF/WDF", "WPC"],
}
MECHANICAL_CHANNEL_MAPPING = {
    "CBC": ["CBC"],
    "CBC+DIFF": ["CBC", "DIFF"],
    "CBC+DIFF+RET": ["CBC", "DIFF", "RET"],
    "CBC+RET": ["CBC", "RET"],
    "CBC+PLT-F": ["CBC", "PLT-F"],
    "CBC+DIFF+PLT-F": ["CBC", "DIFF", "PLT-F"],
    "CBC+DIFF+RET+PLT-F": ["CBC", "DIFF", "RET", "PLT-F"],
    "CBC+RET+PLT-F": ["CBC", "RET", "PLT-F"],
    "CBC+DIFF+WPC": ["CBC", "DIFF", "WPC"],
    "CBC+DIFF+RET+WPC": ["CBC", "DIFF", "RET", "WPC"],
    "CBC+DIFF+PLT-F+WPC": ["CBC", "DIFF", "PLT-F", "WPC"],
    "CBC+DIFF+RET+PLT-F+WPC": ["CBC", "DIFF", "RET", "PLT-F", "WPC"],
    "FREE SELECT": ["CBC"],
}


def read_dictionary():
    df = pd.read_excel(DICTIONARY_PATH, sheet_name=DICTIONARY_SHEET, header=0, engine="odf")
    df = df.dropna(subset=["Feature name RDP A'dam"])
    df.rename(columns={"WDF": "DIFF/WDF"}, inplace=True)
    return df


def build_feature_channel_map(dictionary_df):
    feature_channel_map = {}
    for _, row in dictionary_df.iterrows():
        feature = row["Feature name RDP A'dam"]
        measured = {"YES": [], "OPTION": []}
        for ch in ["CBC-RBC/PLT", "CBC-HGB", "CBC-WNR", "RET", "PLT-F", "WPC", "DIFF/WDF"]:
            val = str(row[ch]).strip().upper()
            if val == "YES":
                measured["YES"].append(ch)
            if val == "OPTION":
                measured["OPTION"].append(ch)
        feature_channel_map[feature] = measured
    return feature_channel_map


def build_discrete_channel_map():
    discrete_channel_map = {}
    for discrete, mech_channels in MECHANICAL_CHANNEL_MAPPING.items():
        discrete_channel_map[discrete] = []
        for mech in mech_channels:
            for ch in APPLICATION_CHANNELS[mech]:
                if ch not in discrete_channel_map[discrete]:
                    discrete_channel_map[discrete].append(ch)
    return discrete_channel_map


def get_cbc_only_features(feature_channel_map):
    cbc_features = []
    for feat, chmap in feature_channel_map.items():
        all_ch = set(chmap["YES"]) | set(chmap["OPTION"])
        if not all_ch:
            continue
        if EXTRA_CHANNELS & all_ch:
            continue  # needs RET / PLT-F / WPC -> out of scope for Model 1
        if all_ch.issubset(CBC_CHANNELS):
            cbc_features.append(feat)
    return cbc_features


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def zero_to_na(row, discrete_channel_map, feature_channel_map, feature_cols):
    discrete = row[DISCRETE_COL]
    if discrete not in discrete_channel_map:
        return row
    channels = discrete_channel_map[discrete]
    for feature in feature_cols:
        if feature not in feature_channel_map:
            continue
        chmap = feature_channel_map[feature]
        if len(chmap["OPTION"]) > 0:
            if not any(ch in channels for ch in chmap["OPTION"]):
                if safe_float(row[feature]) == 0:
                    row[feature] = np.nan
        if len(chmap["YES"]) > 0:
            if feature in RESTRICTED_FEATURES:
                row[feature] = np.nan
            elif not all(ch in channels for ch in chmap["YES"]):
                if safe_float(row[feature]) == 0:
                    row[feature] = np.nan
    return row


def clean_data(df, cbc_features):
    df = df.copy()

    # Replace known placeholder strings with NaN
    df = df.replace(PLACEHOLDER_STRINGS, np.nan)

    # Drop excluded / hs / always-NaN variables up front
    drop_cols = [c for c in (EXCLUDED_VARS + HS_VARS + NAN_ALWAYS_VARS) if c in df.columns]
    df = df.drop(columns=drop_cols)
    cbc_features = [c for c in cbc_features if c not in drop_cols]

    # Drop *_MARK columns entirely: these are Sysmex QC labels, not part of
    # the 159 parameters in Table 1 of the analysis plan.
    mark_cols = [c for c in cbc_features if c.endswith("_MARK")]
    cbc_features = [c for c in cbc_features if c not in mark_cols]

    # Convert CBC measurement columns to numeric (object -> float)
    for col in cbc_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply dictionary-based zero-to-NaN using each row's Discrete value
    dictionary_df = read_dictionary()
    feature_channel_map = build_feature_channel_map(dictionary_df)
    discrete_channel_map = build_discrete_channel_map()
    df = df.apply(
        lambda row: zero_to_na(row, discrete_channel_map, feature_channel_map, cbc_features),
        axis=1,
    )

    # IP flags: missing means "flag not triggered", not genuinely unknown ->
    # fillna(0) rather than statistical imputation (Kasia's convention).
    # Apply to any IP_ABN_*/IP_SUS_* column, not just Kasia's IP_FILL_NA list,
    # since the dictionary export includes newer flags (e.g. Giant_PLT) that
    # her hardcoded list predates but follow the same naming/behaviour.
    ip_flag_cols = [
        c for c in cbc_features
        if c in IP_FILL_NA or c.startswith("IP_ABN_") or c.startswith("IP_SUS_")
    ]
    df[ip_flag_cols] = df[ip_flag_cols].fillna(0)

    # Categorical Q_Flag_* columns: drop from this numeric RF feature set for
    # now (would need one-hot encoding; out of scope for a first pass).
    # Same generalisation as above: any remaining Q_Flag_* column, not just
    # Kasia's 4-item CATEGORICAL_VARS list.
    cbc_features = [
        c for c in cbc_features
        if c not in CATEGORICAL_VARS and not c.startswith("Q_Flag_")
    ]

    return df, cbc_features


def calculate_correlation(X_train, selected_columns, threshold=0.9, verbose=True):
    """Remove one of each pair of features with |Spearman correlation| >
    threshold, protecting PROTECTED_COLS. Adapted from
    fbc_helper_scripts/model_training_methods.py (Kasia)."""
    selected_columns = list(selected_columns)
    correlation_matrix = X_train[selected_columns].corr(method="spearman")
    correlated_features = set()
    names = correlation_matrix.index.tolist()

    for i in range(len(names)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold and names[i] != names[j]:
                col_name, row_name = names[j], names[i]
                if col_name not in PROTECTED_COLS:
                    correlated_features.add(col_name)
                else:
                    correlated_features.add(row_name)

    if verbose:
        print(f"Correlation filter: removing {len(correlated_features)} of "
              f"{len(selected_columns)} features (|Spearman| > {threshold})")
    return [c for c in selected_columns if c not in correlated_features]


def pearson_filter(X, y, selected_columns, p_threshold=0.8, verbose=True):
    """Remove features with weak (p > p_threshold) Pearson association with
    the outcome, protecting PROTECTED_COLS. Adapted from
    fbc_helper_scripts/model_training_methods.py (Kasia)."""
    res = []
    for col in selected_columns:
        sub = pd.concat([X[col], y], axis=1).dropna()
        if sub[col].nunique() <= 1:
            continue
        stat, pval = pearsonr(sub[col], sub[y.name])
        res.append((col, stat, pval))
    res_df = pd.DataFrame(res, columns=["feature", "pearson_stat", "pearson_pval"])

    weak = set(res_df.loc[res_df.pearson_pval > p_threshold, "feature"]) - PROTECTED_COLS
    if verbose:
        print(f"Pearson filter: removing {len(weak)} of {len(selected_columns)} "
              f"features with weak outcome association (p > {p_threshold})")
    return [c for c in selected_columns if c not in weak]


SENSITIVITY_TARGETS = [0.97, 0.98, 0.99]


def calibration_slope_intercept(y_true, p_pred):
    """Calibration-in-the-large: fit y ~ logit(p_pred) via unregularized
    logistic regression. Slope=1, intercept=0 is perfect calibration;
    slope<1 indicates predictions are too extreme (overfitting)."""
    p = np.clip(p_pred, 1e-6, 1 - 1e-6)
    lp = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf)
    lr.fit(lp, y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def find_threshold_for_sensitivity(y_true, p_pred, target_sensitivity):
    """Smallest probability threshold that achieves >= target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, p_pred)
    idx_candidates = np.where(tpr >= target_sensitivity)[0]
    idx = idx_candidates[0] if len(idx_candidates) else len(tpr) - 1
    return thresholds[idx]


def metrics_at_threshold(y_true, p_pred, threshold):
    """PPV, NPV, specificity, sensitivity, and efficiency (TN+FN / N) at a
    given probability threshold, per the analysis plan's performance
    evaluation section."""
    y_pred = (p_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "efficiency": (tn + fn) / len(y_true),
    }


def sensitivity_threshold_metrics(y_true, p_pred, target_sensitivity):
    """Find the probability threshold achieving >= target sensitivity, then
    report PPV, NPV, specificity, and efficiency (TN+FN / N) at that
    threshold, per the analysis plan's performance evaluation section."""
    thr = find_threshold_for_sensitivity(y_true, p_pred, target_sensitivity)
    result = {"target_sensitivity": target_sensitivity, "threshold": thr}
    result.update(metrics_at_threshold(y_true, p_pred, thr))
    return result


def fit_pipeline(X, y, model1_features, rf_params, threshold=0.9, p_threshold=0.8):
    """Run the full model-building procedure (feature selection -> scaling ->
    RF fit) on one dataset, with fixed rf_params. Used both for the apparent
    (full-data) model and for each bootstrap resample, so feature selection
    is re-derived every time rather than reused across resamples."""
    selected = calculate_correlation(X, model1_features, threshold=threshold, verbose=False)
    selected = pearson_filter(X, y, selected, p_threshold=p_threshold, verbose=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[selected])

    model = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced", oob_score=False, **rf_params
    )
    model.fit(X_scaled, y)
    return model, scaler, selected


def bootstrap_optimism(X, y, model1_features, rf_params, n_boot=500, random_state=42):
    """Harrell-style bootstrap optimism correction: repeat the full
    model-building procedure (feature selection + fit) on n_boot bootstrap
    resamples, and for each compare its performance on the resample itself
    (optimistic) against its performance on the original full dataset
    (realistic). The average gap is the estimated optimism -- computed for
    discrimination (AUC), calibration (logistic slope/intercept and Brier
    score), and performance at each fixed sensitivity threshold (specificity,
    PPV, NPV, efficiency) alike, per the analysis plan's "corrected
    c-statistics and calibration estimates" requirement. For the threshold metrics, the
    threshold itself is derived from the resample (matching its own
    sensitivity target) and then applied unchanged to both the resample and
    the original data, consistent with the AUC/calibration optimism logic.

    Hyperparameters are fixed (tuned once beforehand) rather than re-tuned
    per resample -- re-running a full grid search 500x would multiply the
    already multi-minute grid search runtime by ~500x, which is not
    tractable even on this dummy dataset, let alone the full MyDRE cohort.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    records = []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        model, scaler, selected = fit_pipeline(X_boot, y_boot, model1_features, rf_params)

        boot_pred = model.predict_proba(scaler.transform(X_boot[selected]))[:, 1]
        boot_auc = roc_auc_score(y_boot, boot_pred)
        boot_slope, boot_intercept = calibration_slope_intercept(y_boot, boot_pred)
        boot_brier = brier_score_loss(y_boot, boot_pred)

        orig_pred = model.predict_proba(scaler.transform(X[selected]))[:, 1]
        orig_auc = roc_auc_score(y, orig_pred)
        orig_slope, orig_intercept = calibration_slope_intercept(y, orig_pred)
        orig_brier = brier_score_loss(y, orig_pred)

        record = {
            "auc": boot_auc - orig_auc,
            "slope": boot_slope - orig_slope,
            "intercept": boot_intercept - orig_intercept,
            "brier": boot_brier - orig_brier,
        }
        for target in SENSITIVITY_TARGETS:
            thr = find_threshold_for_sensitivity(y_boot, boot_pred, target)
            boot_metrics = metrics_at_threshold(y_boot, boot_pred, thr)
            orig_metrics = metrics_at_threshold(y, orig_pred, thr)
            for key in ("sensitivity", "specificity", "ppv", "npv", "efficiency"):
                record[f"{target}_{key}"] = boot_metrics[key] - orig_metrics[key]
        records.append(record)

        if (b + 1) % 50 == 0:
            running = pd.DataFrame(records).mean()
            print(f"  Bootstrap resample {b + 1}/{n_boot} "
                  f"(running mean optimism: AUC={running['auc']:.4f}, "
                  f"slope={running['slope']:.4f}, intercept={running['intercept']:.4f}, "
                  f"Brier={running['brier']:.4f})")

    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="Path to the cohort CSV (defaults to the synthetic dummy dataset)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=500,
        help="Number of bootstrap resamples for internal validation (default: 500, per analysis plan)",
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
    cbc_features = get_cbc_only_features(feature_channel_map)
    cbc_features = [c for c in cbc_features if c in df.columns]
    print(f"N candidate CBC features from dictionary: {len(cbc_features)}")

    df, cbc_features = clean_data(df, cbc_features)

    # Keep only numeric CBC features for the model (drop remaining flag/MARK
    # columns that are non-numeric / all-NaN after cleaning)
    numeric_cbc_features = [
        c for c in cbc_features
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    print(f"N numeric CBC features after cleaning: {len(numeric_cbc_features)}")

    # Encode sex: Man=1, Vrouw=0, Onbekend -> NaN
    sex_map = {"Man": 1.0, "Vrouw": 0.0}
    df["Geslacht_enc"] = df[SEX_COL].map(sex_map)

    # Encode outcome: Ja=1, Nee=0
    y = (df[OUTCOME_COL] == "Ja").astype(int)

    model1_features = ["Geslacht_enc", AGE_COL, CREATININE_COL] + numeric_cbc_features
    X = df[model1_features]

    print(f"\nModel 1 feature count: {len(model1_features)}")
    print(f"Outcome distribution:\n{y.value_counts()}")
    print("Missingness in model matrix (top 10):")
    print((X.isnull().mean() * 100).sort_values(ascending=False).head(10))

    # CBC is an inclusion criterion, so genuine missingness here should be
    # minimal (mainly: creatinine, and rare true-missing lab values / sex
    # "Onbekend"). No MICE here -- MICE is reserved for the DIFF panel
    # (Model 2/3) per the analysis plan. Drop rows with any remaining gaps
    # so this first pass is a clean complete-case run.
    complete_mask = X.notna().all(axis=1)
    n_dropped = (~complete_mask).sum()
    print(f"\nDropping {n_dropped} rows with residual missing values "
          f"({n_dropped / len(X) * 100:.1f}%) for this complete-case run")
    X = X.loc[complete_mask]
    y = y.loc[complete_mask]

    # Feature selection on the FULL dataset, matching the existing PE pilot's
    # approach (generate_pe_report.py / model_training_methods.py):
    # 1) drop one of each pair of features with |Spearman corr| > 0.9
    # 2) drop features with weak (Pearson p > 0.8) association with outcome
    # No train/test split here -- per the analysis plan, overfitting is
    # instead quantified via bootstrap internal validation below, which uses
    # the full dataset for both model building and evaluation.
    selected_features = calculate_correlation(X, model1_features, threshold=0.9)
    selected_features = pearson_filter(X, y, selected_features, p_threshold=0.8)
    print(f"\nFeatures after correlation+association filtering: "
          f"{len(selected_features)} (was {len(model1_features)})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[selected_features])

    # Hyperparameter search via 5-fold CV (scoring on ROC AUC), reusing the
    # "random_forest_grid_search" grid from model_training_methods.py: it is
    # specifically tuned toward shallow trees / large leaves / aggressive
    # feature subsampling to counter overfitting on datasets this size.
    # Tuned ONCE here on the full data, then held fixed through the bootstrap
    # loop below -- re-running this search inside every one of 500 bootstrap
    # resamples would multiply runtime by ~500x, which is intractable.
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
        base_model, param_grid, cv=5, scoring="roc_auc",
        n_jobs=-1, refit=True, verbose=1,
    )
    grid_search.fit(X_scaled, y)
    model = grid_search.best_estimator_
    rf_params = grid_search.best_params_
    print(f"\nBest params: {rf_params}")
    print(f"Best CV AUC: {grid_search.best_score_:.3f}")

    # Apparent performance: the final model evaluated on the same data it was
    # trained on. This is optimistic (overfitting-inflated) by construction --
    # that's exactly the gap the bootstrap below estimates and corrects for.
    apparent_pred = model.predict_proba(X_scaled)[:, 1]
    apparent_auc = roc_auc_score(y, apparent_pred)
    print(f"Apparent AUC (in-sample, optimistic): {apparent_auc:.3f}")

    importances = pd.Series(model.feature_importances_, index=selected_features)
    print("\nTop 15 features by importance:")
    print(importances.sort_values(ascending=False).head(15))

    # Apparent calibration-in-the-large (logistic slope/intercept of
    # y ~ logit(predicted probability)). Slope=1, intercept=0 is perfect;
    # slope<1 means predictions are too extreme (typical RF overfitting).
    apparent_slope, apparent_intercept = calibration_slope_intercept(y, apparent_pred)
    apparent_brier = brier_score_loss(y, apparent_pred)
    print(f"Apparent calibration slope={apparent_slope:.3f}, "
          f"intercept={apparent_intercept:.3f}")
    print(f"Apparent Brier score (in-sample, optimistic): {apparent_brier:.4f}")

    # Bootstrap internal validation (Harrell-style optimism correction):
    # repeat the full model-building procedure (feature selection + fit,
    # with hyperparameters fixed above) on n_boot bootstrap resamples of the
    # full dataset, and estimate the average optimism -- for AUC and for
    # calibration slope/intercept -- as the gap between each resample's own
    # (optimistic) performance and its performance on the original full
    # dataset.
    print(f"\nRunning bootstrap internal validation ({args.n_bootstrap} resamples)...")
    optimism = bootstrap_optimism(X, y, model1_features, rf_params, n_boot=args.n_bootstrap)
    mean_optimism = optimism.mean()
    corrected_auc = apparent_auc - mean_optimism["auc"]
    corrected_slope = apparent_slope - mean_optimism["slope"]
    corrected_intercept = apparent_intercept - mean_optimism["intercept"]
    corrected_brier = apparent_brier - mean_optimism["brier"]
    print(f"\nMean bootstrap optimism: AUC={mean_optimism['auc']:.4f}, "
          f"slope={mean_optimism['slope']:.4f}, intercept={mean_optimism['intercept']:.4f}, "
          f"Brier={mean_optimism['brier']:.4f}")
    print(f"Optimism-corrected AUC:            {corrected_auc:.3f}")
    print(f"Optimism-corrected cal. slope:     {corrected_slope:.3f}")
    print(f"Optimism-corrected cal. intercept: {corrected_intercept:.3f}")
    print(f"Optimism-corrected Brier score:    {corrected_brier:.4f}")

    # Performance at fixed sensitivity thresholds (97/98/99%), comparable to
    # the 98% sensitivity reported for YEARS -- e.g. "efficiency" is the
    # share of patients (TN+FN) whose CTPA could have been avoided at that
    # threshold. Apparent values are optimistic; corrected values subtract
    # the same bootstrap optimism estimated above, per threshold and metric.
    apparent_rows = [
        sensitivity_threshold_metrics(y, apparent_pred, t) for t in SENSITIVITY_TARGETS
    ]
    apparent_df = pd.DataFrame(apparent_rows).set_index("target_sensitivity")

    metric_keys = ["sensitivity", "specificity", "ppv", "npv", "efficiency"]
    corrected_rows = []
    for target in SENSITIVITY_TARGETS:
        row = {"target_sensitivity": target}
        for key in metric_keys:
            row[key] = apparent_df.loc[target, key] - mean_optimism[f"{target}_{key}"]
        corrected_rows.append(row)
    corrected_threshold_df = pd.DataFrame(corrected_rows).set_index("target_sensitivity")

    print(f"\nPerformance at fixed sensitivity thresholds (apparent, uncorrected):")
    print(apparent_df[["threshold"] + metric_keys].round(3))
    print(f"\nPerformance at fixed sensitivity thresholds "
          f"(bootstrap-corrected, n={args.n_bootstrap} resamples):")
    print(corrected_threshold_df.round(3))

    # ROC curve of the apparent (full-data) model. The bootstrap correction
    # is a scalar shift in AUC, not a separate curve, so it's reported in the
    # title alongside the (optimistic) apparent curve.
    fpr, tpr, _ = roc_curve(y, apparent_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Apparent (AUC={apparent_auc:.3f})", color="tab:blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(
        "Model 1 (age + sex + creatinine + CBC) - ROC curve\n"
        f"Bootstrap-corrected AUC={corrected_auc:.3f} "
        f"(n={args.n_bootstrap} resamples, optimism={mean_optimism['auc']:.3f})"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    roc_out_path = REPO_ROOT / "model1_cbc_roc_curve.png"
    fig.savefig(roc_out_path, dpi=150)
    print(f"\nROC curve saved to: {roc_out_path}")

    # Calibration plot: observed vs predicted probability, in deciles of
    # predicted risk, plus the bootstrap-corrected calibration line.
    obs_freq, pred_freq = calibration_curve(y, apparent_pred, n_bins=10, strategy="quantile")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(pred_freq, obs_freq, marker="o", label="Apparent (observed vs predicted)",
              color="tab:blue")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "Model 1 (age + sex + creatinine + CBC) - Calibration plot\n"
        f"Bootstrap-corrected slope={corrected_slope:.3f}, "
        f"intercept={corrected_intercept:.3f} (n={args.n_bootstrap} resamples)"
    )
    ax2.legend(loc="upper left")
    fig2.tight_layout()

    cal_out_path = REPO_ROOT / "model1_cbc_calibration_curve.png"
    fig2.savefig(cal_out_path, dpi=150)
    print(f"Calibration curve saved to: {cal_out_path}")


if __name__ == "__main__":
    main()
