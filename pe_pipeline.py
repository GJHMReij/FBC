"""
PE prediction pipeline following the 80/20 temporal-split analysis plan.

Per model (1 = CBC, 2 = CBC+DIFF, 3 = CBC+DIFF+D-dimer):
  1. Load cohort, apply definitive outcome, exclude outpatients, parse scan date.
  2. Temporal split: hold-out test set = exactly the most recent 20% of scans,
     built per patient (a patient is wholly in one set; patients with scans in
     both periods are allocated entirely to the training set).
  3. MICE, separately for training and test set (outcome NOT in the imputation
     model): 10 imputation sets each.
  4. Grid search (5-fold, patient-grouped CV, AUC) on training imputation set 1.
  5. Build one RF per training imputation set; the mean predicted probability
     of the 10 RFs is the model.
  6. Evaluate on the 10 imputed test sets; per-imputation estimates are pooled
     with Rubin's rules (within-imputation variance from a patient-level
     bootstrap of the test set). Training ("apparent") performance is reported
     the same way.

Fixed-sensitivity operating points (97/98/99%, plus 95% for Table 2):
  A = threshold chosen on the TRAINING set (out-of-bag predictions) and applied
      unchanged to the test set -> the sensitivity actually achieved is reported.
  B = threshold chosen on the evaluated set itself (sensitivity fixed by
      construction).

Run on MyDRE (real data) via run_model1.py / run_model2.py / run_model3.py.
Local smoke test on the dummy CSV (its dates are placeholders):
    .venv/bin/python pe_pipeline.py --models 1 --quick --fake-dates-for-testing
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from pe_model1_cbc_rf import (
    AGE_COL, CREATININE_COL, DEFAULT_INPUT_CSV, NOT_ASSESSABLE_VALUE, ORDER_ID_COL,
    OUTCOME_COL, OUTCOME_CORRECTION_COL, REPO_ROOT, SEX_COL, Heartbeat,
    apply_outcome_correction, build_feature_channel_map, calibration_slope_intercept,
    clean_data, get_cbc_only_features, read_dictionary, save_manuscript_data,
    save_roc_data,
)
from pe_model2_cbc_diff_rf import (
    D_DIMER_ASSAY_COL, D_DIMER_ASSAY_MAP, D_DIMER_VALUE_COL, HOSPITAL_COL,
    get_cbc_diff_features, rubin_pool,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

DATE_COL = "Startmoment_Beeldvorming"
PATIENT_COL = "Pseudo_id"
CLASS_COL = "Patientklasse"
EXCLUDE_CLASS_PATTERN = r"^\s*poliklinisch\s*$"
SENS_TARGETS = [0.95, 0.97, 0.98, 0.99]
METRIC_KEYS = ["sensitivity", "specificity", "ppv", "npv", "efficiency", "accuracy", "f1"]
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M", "%Y-%m-%d", "%d-%m-%Y",
]
MODEL_LABELS = {1: "Model 1 (CBC)", 2: "Model 2 (+DIFF)", 3: "Model 3 (+D-dimer)"}

# Grid according to Kasia (identical for all models)
FULL_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "min_samples_leaf": [10, 25, 50],
    "min_samples_split": [20, 50, 100],
    "max_features": ["sqrt", 0.2, 0.3],
}
QUICK_GRID = {
    "n_estimators": [100],
    "max_depth": [5, 9],
    "min_samples_leaf": [10, 25],
    "min_samples_split": [20],
    "max_features": ["sqrt"],
}


# --------------------------------------------------------------------------- data

def parse_dates(series):
    """Parse scan timestamps, trying the same formats as the R check; keeps the
    format that parses the most values."""
    s = series.astype(str).str.strip()
    best = None
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        n_ok = int(parsed.notna().sum())
        if best is None or n_ok > best[1]:
            best = (parsed, n_ok, fmt)
    if best[1] < 0.99 * len(s):
        try:
            generic = pd.to_datetime(s, errors="coerce")
            if int(generic.notna().sum()) > best[1]:
                best = (generic, int(generic.notna().sum()), "pandas-inferred")
        except Exception:
            pass
    return best


def patient_ids(df):
    if PATIENT_COL not in df.columns:
        print(f"WARNING: '{PATIENT_COL}' not found -- every scan treated as its own patient")
        return pd.Series([f"row_{i}" for i in df.index], index=df.index)
    pid = df[PATIENT_COL].astype(str)
    na = df[PATIENT_COL].isna()
    pid[na] = [f"missing_{i}" for i in df.index[na]]
    return pid


def temporal_split(dates, patients, test_fraction):
    """Hold-out test set = exactly `test_fraction` of all scans, built patient by
    patient from the most recent first-scan dates backwards. A patient is always
    wholly in one set, so the test set only contains patients without earlier
    scans; patients with scans in both periods end up entirely in the training
    set. A patient whose scans would overshoot the target is skipped and the next
    most recent patient that fits is taken, so the target is hit exactly."""
    n = len(dates)
    target = int(round(test_fraction * n))
    pat = pd.DataFrame({"pid": patients.values, "date": dates.values})
    g = pat.groupby("pid").agg(first=("date", "min"), n=("date", "size"))
    g = g.sort_values("first", ascending=False)
    chosen, total, skipped = [], 0, 0
    for pid, first, cnt in zip(g.index, g["first"], g["n"]):
        if total == target:
            break
        if total + cnt <= target:
            chosen.append(pid)
            total += cnt
        else:
            skipped += 1
    is_test = patients.isin(set(chosen)).to_numpy()
    cutoff = pd.Timestamp(g.loc[chosen, "first"].min())
    late_training = (~is_test) & (dates.values > np.datetime64(cutoff))
    n_both = patients[late_training].nunique()
    if total != target:
        print(f"NOTE: test set is {total} scans, target was {target}")
    print(f"Test set built from {len(chosen)} patients; {skipped} multi-scan patient(s) "
          f"skipped at the boundary to hit the target exactly")
    return pd.Series(is_test, index=dates.index), cutoff, n_both, int(late_training.sum())


def load_and_prepare(args):
    print(f"Loading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Shape: {df.shape}")

    if args.outcome_correction is not None:
        print(f"\nApplying outcome correction from {args.outcome_correction} ...")
        df = apply_outcome_correction(df, args.outcome_correction)
    df = df.reset_index(drop=True)

    outcome_ok = df[OUTCOME_COL].isin(["Ja", "Nee"])
    if (~outcome_ok).any():
        print(f"Dropping {int((~outcome_ok).sum())} scans without a Ja/Nee outcome")
        df = df.loc[outcome_ok].reset_index(drop=True)
    print(f"Scans after outcome handling: {len(df)} "
          f"({int((df[OUTCOME_COL] == 'Ja').sum())} PE, "
          f"{(df[OUTCOME_COL] == 'Ja').mean() * 100:.1f}%)")

    if CLASS_COL in df.columns:
        poli = df[CLASS_COL].astype(str).str.match(EXCLUDE_CLASS_PATTERN, case=False)
        print(f"Excluding {int(poli.sum())} outpatient scans ({CLASS_COL} = Poliklinisch)")
        df = df.loc[~poli].reset_index(drop=True)
    else:
        print(f"WARNING: '{CLASS_COL}' not found -- no outpatient exclusion applied")

    if args.require_ddimer or args.models == [3]:
        has_dd = pd.to_numeric(df[D_DIMER_VALUE_COL], errors="coerce").notna()
        print(f"Excluding {int((~has_dd).sum())} scans without a measured D-dimer "
              f"({D_DIMER_VALUE_COL}) BEFORE the split; {int(has_dd.sum())} scans remain")
        df = df.loc[has_dd].reset_index(drop=True)

    parsed, n_ok, fmt = parse_dates(df[DATE_COL])
    print(f"Scan date ({DATE_COL}): format {fmt}, {n_ok} of {len(df)} valid")
    if n_ok == 0 and args.fake_dates_for_testing:
        print("TEST MODE: placeholder dates replaced by synthetic dates and ~5% repeated patients")
        rng = np.random.default_rng(0)
        parsed = pd.Series(pd.Timestamp("2017-01-01")
                           + pd.to_timedelta(rng.integers(0, 2700, len(df)), unit="D"), index=df.index)
        ids = np.arange(len(df)).astype(str)
        dup = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
        ids[dup] = ids[rng.integers(0, len(df), len(dup))]
        df[PATIENT_COL] = ids
    elif n_ok < 0.99 * len(df):
        raise ValueError(f"Only {n_ok} of {len(df)} scan dates could be parsed -- check the date format")
    df["scan_date"] = parsed
    bad_date = df["scan_date"].isna()
    if bad_date.any():
        print(f"Dropping {int(bad_date.sum())} scans without a valid date")
        df = df.loc[~bad_date].reset_index(drop=True)

    df["sex_enc"] = df[SEX_COL].map({"Man": 1.0, "Vrouw": 0.0})
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[CREATININE_COL] = pd.to_numeric(df[CREATININE_COL], errors="coerce")
    bad = df["sex_enc"].isna() | df[AGE_COL].isna()
    if bad.any():
        print(f"Dropping {int(bad.sum())} scans with missing/unknown sex or age")
        df = df.loc[~bad].reset_index(drop=True)

    # data cleaning (zero -> missing per data dictionary, drop excluded variables)
    feature_channel_map = build_feature_channel_map(read_dictionary())
    cbc_only = set(get_cbc_only_features(feature_channel_map))
    candidates = [c for c in get_cbc_diff_features(feature_channel_map) if c in df.columns]
    print(f"Candidate CBC+DIFF features from dictionary: {len(candidates)}")
    df, cleaned = clean_data(df, candidates)
    numeric = [c for c in cleaned if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    cbc_numeric = [c for c in numeric if c in cbc_only]
    print(f"Numeric features after cleaning: CBC {len(cbc_numeric)}, CBC+DIFF {len(numeric)}")

    df[D_DIMER_VALUE_COL] = pd.to_numeric(df[D_DIMER_VALUE_COL], errors="coerce")
    df["D_dimer_assay_enc"] = df[D_DIMER_ASSAY_COL].map(D_DIMER_ASSAY_MAP)
    df["calendar_year"] = df["scan_date"].dt.year.astype(float)

    aux_cols = [D_DIMER_VALUE_COL, "D_dimer_assay_enc", "calendar_year"]
    if HOSPITAL_COL in df.columns and df[HOSPITAL_COL].notna().any():
        hosp = pd.get_dummies(df[HOSPITAL_COL], prefix="hosp", dtype=float)
        df = pd.concat([df, hosp], axis=1)
        aux_cols += list(hosp.columns)

    features = {
        1: ["sex_enc", AGE_COL, CREATININE_COL] + cbc_numeric,
        2: ["sex_enc", AGE_COL, CREATININE_COL] + numeric,
        3: ["sex_enc", AGE_COL, CREATININE_COL, D_DIMER_VALUE_COL, "D_dimer_assay_enc"] + numeric,
    }
    imputation_cols = ["sex_enc", AGE_COL, CREATININE_COL] + numeric + aux_cols
    all_nan = [c for c in imputation_cols if df[c].isna().all()]
    if all_nan:
        raise ValueError(f"Columns 100% missing, cannot be used/imputed: {all_nan}")
    y = (df[OUTCOME_COL] == "Ja").astype(int)
    return df, y, features, imputation_cols


# ------------------------------------------------------------------------ imputation

def impute_sets(frame, train_mask, test_mask, n_imputations, max_iter, seed, cache_dir, tag):
    """MICE, run separately on the training and the test set (no outcome in the
    imputation model). Cached on disk so Models 1 and 2 share one run."""
    key_src = (tag, tuple(frame.index[train_mask]), tuple(frame.index[test_mask]),
               tuple(frame.columns), n_imputations, max_iter, seed)
    key = hashlib.md5(repr(key_src).encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"imputed_{tag}_{key}.joblib"
    if cache_path.exists():
        print(f"Loading cached imputation sets: {cache_path.name}")
        return joblib.load(cache_path)

    train_frame = frame.loc[train_mask]
    test_frame = frame.loc[test_mask].copy()
    cols = [c for c in frame.columns if train_frame[c].notna().any()]
    for c in cols:
        if test_frame[c].isna().all():
            print(f"WARNING: '{c}' is empty in the test set; filled with the training median")
            test_frame[c] = train_frame[c].median()
    train_frame, test_frame = train_frame[cols], test_frame[cols]

    def run(source, offset, label):
        sets = []
        for m in range(n_imputations):
            imputer = IterativeImputer(sample_posterior=True, random_state=seed + offset + m,
                                       max_iter=max_iter, initial_strategy="median")
            with Heartbeat(f"MICE {label} {m + 1}/{n_imputations}"):
                imputed = imputer.fit_transform(source)
            sets.append(pd.DataFrame(imputed, columns=cols, index=source.index))
            print(f"  MICE {label} imputation {m + 1}/{n_imputations} done", flush=True)
        return sets

    train_sets = run(train_frame, 0, "training")
    test_sets = run(test_frame, 1000, "test")
    result = {"train": train_sets, "test": test_sets}
    joblib.dump(result, cache_path)
    return result


# ---------------------------------------------------------------------- performance

def thr_for_sens(y, p, target):
    """Largest threshold such that sensitivity >= target."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return pos[min(k, len(pos) - 1)]


def metrics_np(y, p, thr):
    pred = p >= thr
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    n = len(y)

    def div(a, b):
        return a / b if b else np.nan
    return {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "efficiency": (tn + fn) / n, "accuracy": (tp + tn) / n,
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def stat_vector(y, p, thr_train=None):
    out = {"auc": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2))}
    try:
        slope, intercept = calibration_slope_intercept(y, p)
    except Exception:
        slope, intercept = np.nan, np.nan
    out["slope"], out["intercept"] = slope, intercept
    for t in SENS_TARGETS:
        for k, v in metrics_np(y, p, thr_for_sens(y, p, t)).items():
            out[f"B{t}_{k}"] = v
        if thr_train is not None:
            for k, v in metrics_np(y, p, thr_train[t]).items():
                out[f"A{t}_{k}"] = v
    return out


def make_draws(groups, n_draws, rng):
    """Patient-level bootstrap draws (indices of scans)."""
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    members = np.split(order, np.cumsum(counts)[:-1])
    n_pat = len(members)
    return [np.concatenate([members[i] for i in rng.integers(0, n_pat, n_pat)])
            for _ in range(n_draws)]


def pooled_performance(y, p_list, thr_list, draws):
    """Per imputation: point estimate + bootstrap variance; then Rubin's rules."""
    per_m_est, per_m_var = [], []
    for p, thr in zip(p_list, thr_list):
        point = stat_vector(y, p, thr)
        boots = [stat_vector(y[idx], p[idx], thr) for idx in draws]
        keys = list(point)
        var = {k: float(np.nanvar([b[k] for b in boots], ddof=1)) if len(boots) > 1 else 0.0
               for k in keys}
        per_m_est.append(point)
        per_m_var.append(var)
    pooled = {}
    for k in per_m_est[0]:
        est, _, se = rubin_pool([e[k] for e in per_m_est], [v[k] for v in per_m_var])
        lo, hi = est - 1.96 * se, est + 1.96 * se
        if k not in ("slope", "intercept"):
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        pooled[k] = {"est": float(est), "se": float(se), "lo": float(lo), "hi": float(hi)}
    return pooled


def pooled_table(pooled, prefix):
    rows = {}
    for t in SENS_TARGETS:
        row = {}
        for k in METRIC_KEYS:
            d = pooled.get(f"{prefix}{t}_{k}")
            row[k] = f"{d['est']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" if d else ""
        rows[f"{int(t * 100)}%"] = row
    return pd.DataFrame(rows).T


def ensemble_summary(y, p, thr_train, draws):
    point = stat_vector(y, p, thr_train)
    boots = [stat_vector(y[idx], p[idx], thr_train) for idx in draws]
    out = {}
    for k, v in point.items():
        vals = np.array([b[k] for b in boots], dtype=float)
        out[k] = {"est": float(v), "lo": float(np.nanpercentile(vals, 2.5)),
                  "hi": float(np.nanpercentile(vals, 97.5))}
    return out


# ---------------------------------------------------------------------------- model

def run_model(model_no, args, df, y, features, imputation_cols, is_test, cutoff, out_dir):
    label = MODEL_LABELS[model_no]
    feats = features[model_no]
    rng = np.random.default_rng(args.seed + model_no)
    print(f"\n{'=' * 70}\n{label}: {len(feats)} features\n{'=' * 70}")

    pop = pd.Series(True, index=df.index)
    if model_no == 3:
        pop = df[D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna()
        print(f"Excluding {int((~pop).sum())} scans without a measured D-dimer "
              f"(not imputed), in training and test set")
    train_mask = (pop & ~is_test).to_numpy()
    test_mask = (pop & is_test).to_numpy()
    y_train, y_test = y[train_mask].to_numpy(), y[test_mask].to_numpy()
    g_train = patient_ids(df)[train_mask].to_numpy()
    g_test = patient_ids(df)[test_mask].to_numpy()
    hosp_test = df.loc[test_mask, HOSPITAL_COL].to_numpy() if HOSPITAL_COL in df.columns else None
    print(f"Training: {train_mask.sum()} scans ({y_train.sum()} PE, {y_train.mean() * 100:.1f}%)")
    print(f"Hold-out test: {test_mask.sum()} scans ({y_test.sum()} PE, {y_test.mean() * 100:.1f}%)")

    tag = "pop_dd" if (model_no == 3 or args.require_ddimer) else "pop_all"
    frame = df[imputation_cols]
    sets = impute_sets(frame, train_mask, test_mask, args.n_imputations, args.mice_max_iter,
                       args.seed, out_dir / "cache", tag)
    train_sets, test_sets = sets["train"], sets["test"]
    missing_cols = [f for f in feats if f not in train_sets[0].columns]
    if missing_cols:
        raise ValueError(f"Model features lost during imputation: {missing_cols}")

    grid = QUICK_GRID if args.quick else FULL_GRID
    n_combo = int(np.prod([len(v) for v in grid.values()]))
    print(f"\nGrid search: {n_combo} combinations x 5 folds (patient-grouped, AUC) "
          f"on training imputation set 1 ...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = GridSearchCV(RandomForestClassifier(random_state=args.seed, n_jobs=1),
                          grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=False, verbose=0)
    with Heartbeat("GridSearchCV"):
        search.fit(train_sets[0][feats], y_train, groups=g_train)
    best = search.best_params_
    print(f"Best params (fixed for all imputation sets): {best}")
    print(f"Best CV AUC (imputation set 1): {search.best_score_:.3f}")

    print(f"\nBuilding {args.n_imputations} random forests ...")
    forests, p_train, p_oob, p_test = [], [], [], []
    for m in range(args.n_imputations):
        rf = RandomForestClassifier(**best, oob_score=True,
                                    random_state=args.seed + m, n_jobs=-1)
        with Heartbeat(f"RF {m + 1}"):
            rf.fit(train_sets[m][feats], y_train)
        oob = rf.oob_decision_function_[:, 1]
        oob = np.where(np.isnan(oob), np.nanmean(oob), oob)
        forests.append(rf)
        p_train.append(rf.predict_proba(train_sets[m][feats])[:, 1])
        p_oob.append(oob)
        p_test.append(rf.predict_proba(test_sets[m][feats])[:, 1])
        print(f"  RF {m + 1}/{args.n_imputations}: training AUC {roc_auc_score(y_train, p_train[-1]):.3f}, "
              f"test AUC {roc_auc_score(y_test, p_test[-1]):.3f}", flush=True)

    # thresholds for the fixed sensitivities, from the training set (OOB predictions)
    thr_train_m = [{t: thr_for_sens(y_train, o, t) for t in SENS_TARGETS} for o in p_oob]
    p_ens_train = np.mean(p_train, axis=0)
    p_ens_oob = np.mean(p_oob, axis=0)
    p_ens_test = np.mean(p_test, axis=0)
    thr_ens = {t: thr_for_sens(y_train, p_ens_oob, t) for t in SENS_TARGETS}

    print(f"\nBootstrap ({args.n_bootstrap} patient-level draws on the test set; "
          f"{args.n_bootstrap_train} on the training set) + Rubin's rules ...")
    draws_test = make_draws(g_test, args.n_bootstrap, rng)
    draws_train = make_draws(g_train, args.n_bootstrap_train, rng)
    with Heartbeat("bootstrap"):
        pooled_test = pooled_performance(y_test, p_test, thr_train_m, draws_test)
        pooled_train = pooled_performance(y_train, p_train, [None] * len(p_train), draws_train)
        ens_test = ensemble_summary(y_test, p_ens_test, thr_ens, draws_test)

    def line(d):
        return f"{d['est']:.3f} (95% CI {d['lo']:.3f}-{d['hi']:.3f})"

    print(f"\n--- {label}: hold-out test set (Rubin-pooled over {args.n_imputations} imputations) ---")
    for k, name in [("auc", "AUC"), ("brier", "Brier"), ("slope", "Calibration slope"),
                    ("intercept", "Calibration intercept")]:
        print(f"{name}: {line(pooled_test[k])}")
    print("\nOperating points A: threshold from TRAINING set (OOB), applied to the test set")
    tbl_a = pooled_table(pooled_test, "A")
    print(tbl_a.to_string())
    print("\nOperating points B: sensitivity fixed on the test set itself")
    tbl_b = pooled_table(pooled_test, "B")
    print(tbl_b.to_string())
    print(f"\n--- {label}: training set, apparent (Rubin-pooled) ---")
    print(f"AUC: {line(pooled_train['auc'])}")
    tbl_train = pooled_table(pooled_train, "B")
    print(tbl_train.to_string())
    print(f"\n--- {label}: ensemble (mean of the {args.n_imputations} forests), test set ---")
    print(f"AUC: {line(ens_test['auc'])}")

    per_hospital = {}
    if hosp_test is not None:
        print("\nTest set per hospital (ensemble; thresholds from training):")
        for h in sorted(pd.Series(hosp_test).dropna().unique()):
            mk = hosp_test == h
            if mk.sum() < 30 or y_test[mk].sum() < 5:
                continue
            sub = ensemble_summary(y_test[mk], p_ens_test[mk], thr_ens,
                                   make_draws(g_test[mk], min(args.n_bootstrap, 200), rng))
            per_hospital[str(h)] = sub
            print(f"  {h}: n={int(mk.sum())}, PE={y_test[mk].mean() * 100:.1f}%, "
                  f"AUC {line(sub['auc'])}, "
                  f"95%-target sens {sub['A0.95_sensitivity']['est']:.3f}, "
                  f"NPV {sub['A0.95_npv']['est']:.3f}, eff {sub['A0.95_efficiency']['est']:.3f}")

    importance = pd.Series(np.mean([rf.feature_importances_ for rf in forests], axis=0),
                           index=feats).sort_values(ascending=False)
    print("\nTop 15 features (mean importance over the forests, training):")
    print(importance.head(15).round(4).to_string())

    # outputs ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = out_dir / f"model{model_no}"
    fpr_te, tpr_te, _ = roc_curve(y_test, p_ens_test)
    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_ens_train)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_tr, tpr_tr, color="tab:blue", alpha=0.8,
            label=f"Training, apparent (AUC={roc_auc_score(y_train, p_ens_train):.3f})")
    ax.plot(fpr_te, tpr_te, color="tab:orange",
            label=f"Hold-out test (AUC={pooled_test['auc']['est']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{label} - ROC curve\nTest AUC {line(pooled_test['auc'])}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{pre}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    obs, pred = calibration_curve(y_test, p_ens_test, n_bins=10, strategy="quantile")
    # continuous calibration line over 0-1: observed = expit(intercept + slope * logit(p))
    xs = np.linspace(0.001, 0.999, 200)
    cal_line = 1 / (1 + np.exp(-(ens_test["intercept"]["est"]
                                 + ens_test["slope"]["est"] * np.log(xs / (1 - xs)))))
    fig2, (ax2, axh) = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True,
                                    gridspec_kw={"height_ratios": [4, 1]})
    ax2.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax2.plot(xs, cal_line, color="tab:orange",
             label=f"Calibration line (slope={ens_test['slope']['est']:.3f}, "
                   f"intercept={ens_test['intercept']['est']:.3f})")
    ax2.plot(pred, obs, marker="o", linestyle="none", color="tab:blue", label="Deciles of predicted risk")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Observed frequency")
    ax2.set_title(f"{label} - Calibration plot (hold-out test set)")
    ax2.legend(loc="upper left")
    axh.hist(p_ens_test, bins=np.linspace(0, 1, 51), color="tab:grey")
    axh.set_xlabel("Predicted probability")
    axh.set_ylabel("Scans")
    fig2.tight_layout()
    fig2.savefig(f"{pre}_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    importance.to_csv(f"{pre}_feature_importance.csv", header=["importance"])
    tbl_a.to_csv(f"{pre}_test_thresholds_from_training.csv")
    tbl_b.to_csv(f"{pre}_test_thresholds_on_test.csv")
    tbl_train.to_csv(f"{pre}_training_apparent.csv")
    results = {
        "label": label, "n_features": len(feats), "best_params": best,
        "best_cv_auc": float(search.best_score_), "cutoff_date": str(cutoff.date()),
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "pe_train": int(y_train.sum()), "pe_test": int(y_test.sum()),
        "test_pooled": pooled_test, "train_pooled": pooled_train,
        "ensemble_test": ens_test, "ensemble_thresholds": {str(k): float(v) for k, v in thr_ens.items()},
        "per_hospital": per_hospital,
    }
    with open(f"{pre}_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    save_roc_data(label, fpr_te, tpr_te, pooled_test["auc"]["est"], f"{pre}_roc_data.json")
    m95 = {k: pooled_test[f"A0.95_{k}"]["est"] for k in ("sensitivity", "specificity", "accuracy", "f1")}
    save_manuscript_data(label, pooled_test["auc"]["est"], pooled_test["brier"]["est"], m95,
                         list(importance.head(10).items()), f"{pre}_manuscript_data.json")
    joblib.dump({"label": label, "features": feats, "members": forests,
                 "thresholds_ensemble": thr_ens, "best_params": best,
                 "cutoff_date": str(cutoff.date())}, f"{pre}_ensemble.joblib")
    # row-level predictions: lets later scripts draw all models in one figure
    pd.concat([
        pd.DataFrame({"set": "train", "y": y_train, "p_ensemble": p_ens_train}),
        pd.DataFrame({"set": "test", "y": y_test, "p_ensemble": p_ens_test,
                      "hospital": hosp_test if hosp_test is not None else np.nan}),
    ], ignore_index=True).to_csv(f"{pre}_predictions.csv", index=False)

    try:
        from pe_report import build_report
        hosp_counts = lambda m: (df.loc[m, HOSPITAL_COL].value_counts().to_dict()
                                 if HOSPITAL_COL in df.columns else {})
        info = {
            "label": label,
            "intro": ("Random Forest ensemble (mean of one forest per imputed data set), temporal "
                      "80/20 split; hold-out test = most recent scans; Rubin's rules over the imputations."),
            "split": [
                f"Training: {int(train_mask.sum())} scans, {int(y_train.sum())} PE ({y_train.mean() * 100:.1f}%); hospitals {hosp_counts(train_mask)}",
                f"Hold-out test: {int(test_mask.sum())} scans, {int(y_test.sum())} PE ({y_test.mean() * 100:.1f}%); hospitals {hosp_counts(test_mask)}",
                f"Test patients' first scan on/after {cutoff.date()}; patients with scans in both periods are in the training set",
                "Excluded: outpatient scans, scans without definitive outcome" + ("; scans without D-dimer" if model_no == 3 else ""),
            ],
            "settings": [
                ("Number of features", len(feats)),
                ("Imputations (training and test, separately)", args.n_imputations),
                ("MICE iterations", args.mice_max_iter),
                ("Best hyperparameters (grid search, 5-fold CV, AUC)", best),
                ("Best cross-validated AUC", f"{float(search.best_score_):.3f}"),
                ("Bootstrap draws (test / training)", f"{args.n_bootstrap} / {args.n_bootstrap_train}"),
                ("Quick (smoke-test) mode", "YES - not final numbers" if args.quick else "no"),
            ],
            "test_metrics": [
                ("AUC (test, pooled)", line(pooled_test["auc"])),
                ("AUC (test, ensemble)", line(ens_test["auc"])),
                ("AUC (training, apparent)", line(pooled_train["auc"])),
                ("Brier score (test)", line(pooled_test["brier"])),
                ("Calibration slope (test)", line(pooled_test["slope"])),
                ("Calibration intercept (test)", line(pooled_test["intercept"])),
            ],
            "images": [("ROC curve", f"{pre}_roc_curve.png"),
                       ("Calibration plot (hold-out test set)", f"{pre}_calibration_curve.png")],
            "tables": [
                ("Test set: threshold chosen on the training set", tbl_a,
                 "Threshold for the target sensitivity determined on training OOB predictions, applied unchanged to the test set."),
                ("Test set: sensitivity fixed on the test set", tbl_b,
                 "Threshold chosen on the test set itself so that sensitivity equals the target."),
                ("Training set (apparent)", tbl_train, "Optimistic by construction."),
            ],
            "hospitals": ([["Hospital", "n", "PE %", "AUC", "Sens (95% target)", "NPV", "Efficiency"]] + [
                [h, int((hosp_test == h).sum()), f"{y_test[hosp_test == h].mean() * 100:.1f}", line(v["auc"]),
                 f"{v['A0.95_sensitivity']['est']:.3f}", f"{v['A0.95_npv']['est']:.3f}",
                 f"{v['A0.95_efficiency']['est']:.3f}"] for h, v in per_hospital.items()]) if per_hospital else None,
            "importance": importance,
        }
        report_path = build_report(pre, info)
        print(f"Word report: {report_path}")
    except Exception as exc:  # the report must never break a finished run
        print(f"WARNING: could not write the Word report: {exc!r}")

    print(f"\nOutputs written to {out_dir} (prefix model{model_no}_)")


# -------------------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--outcome-correction", type=Path, default=None,
                   help=f"Excel/CSV with the definitive outcome ('{OUTCOME_CORRECTION_COL}', "
                        f"joined on '{ORDER_ID_COL}'); rows '{NOT_ASSESSABLE_VALUE}' are dropped")
    p.add_argument("--models", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--n-imputations", type=int, default=10)
    p.add_argument("--mice-max-iter", type=int, default=10)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--n-bootstrap-train", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results_temporal")
    p.add_argument("--require-ddimer", action="store_true",
                   help="exclude scans without a measured D-dimer before the split (automatic when running only Model 3)")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: 2 imputations, 2 MICE iterations, tiny grid, 20 bootstrap draws")
    p.add_argument("--fake-dates-for-testing", action="store_true",
                   help="Dummy CSV only: its dates are placeholders, so synthesise dates and repeated patients")
    args = p.parse_args()
    if args.quick:
        args.n_imputations = min(args.n_imputations, 2)
        args.mice_max_iter = min(args.mice_max_iter, 2)
        args.n_bootstrap = min(args.n_bootstrap, 20)
        args.n_bootstrap_train = min(args.n_bootstrap_train, 10)
    return args


def main():
    args = parse_args()
    t0 = time.time()
    if args.quick:
        print("QUICK MODE: reduced imputations/iterations/grid/bootstrap -- smoke test only, "
              "numbers are not final results")
    df, y, features, imputation_cols = load_and_prepare(args)

    patients = patient_ids(df)
    is_test, cutoff, n_both, n_moved = temporal_split(df["scan_date"], patients, args.test_fraction)
    print(f"\nTemporal split: test = exactly {args.test_fraction * 100:.0f}% of {len(df)} scans "
          f"-> training {int((~is_test).sum())} scans, hold-out test {int(is_test.sum())} scans")
    print(f"First scan of the most recent test patients: {cutoff.date()}; test scans span "
          f"{df.loc[is_test, 'scan_date'].min().date()} to {df.loc[is_test, 'scan_date'].max().date()}")
    print(f"Patients with scans in both periods: {n_both} "
          f"({n_moved} training scans dated after the cut-off; patient wholly in training)")
    for name, mask in [("training", ~is_test), ("test", is_test)]:
        extra = ""
        if HOSPITAL_COL in df.columns:
            extra = f", hospital {df.loc[mask, HOSPITAL_COL].value_counts().to_dict()}"
        print(f"  {name}: {int(mask.sum())} scans, PE {y[mask].mean() * 100:.1f}%{extra}")

    for model_no in args.models:
        run_model(model_no, args, df, y, features, imputation_cols, is_test, cutoff, args.out_dir)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
