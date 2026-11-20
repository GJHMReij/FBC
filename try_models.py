"""
Quick check whether other models / stronger regularisation / variable selection give a higher test
AUC than the current Random Forest, on the SAME temporal split and the SAME cached MICE sets as the
main pipeline (use the same --out-dir, --outcome-in-train-imputation and --seed so the cache is hit).

Candidates (one model number per run):
  rf_current     RF with the hyperparameters found by the main run (BASELINE_PARAMS below)
  rf_regularised RF, grid with shallower trees / larger leaves / fewer features per split
  elastic_net    logistic regression, elastic-net penalty (l1_ratio 0 = ridge ... 1 = LASSO), standardised
  lasso          logistic regression, L1 penalty only
  xgboost        XGBoost (HistGradientBoosting if xgboost is not installed)
  rf_on_lasso    LASSO selects variables (per training set), RF (current params) on the selected ones

Tuning: 5-fold patient-grouped CV on AUC, on the first 3 imputed training sets (as in the main run).
Then one fit per imputed training set, evaluated on its own imputed test set.
Output: console table + try_models_m<N>.csv in --out-dir (aggregates only, no patient data).

Run on MyDRE, e.g.:
    python try_models.py --models 1 --input-csv ... --outcome-correction ... \
        --outcome-in-train-imputation --tuning-sets 3 --out-dir results_temporal_outcome_imp
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pe_pipeline as pp
from pe_model1_cbc_rf import Heartbeat
from pe_model2_cbc_diff_rf import D_DIMER_VALUE_COL, HOSPITAL_COL  # noqa: F401

# best hyperparameters of the main run (outcome-in-imputation run, results_temporal_outcome_imp)
BASELINE_PARAMS = {
    1: {"max_depth": 9, "max_features": 0.3, "min_samples_leaf": 25, "min_samples_split": 100},
    2: {"max_depth": 9, "max_features": 0.4, "min_samples_leaf": 50, "min_samples_split": 10},
    3: {"max_depth": 9, "max_features": 0.4, "min_samples_leaf": 50, "min_samples_split": 10},
    4: {"max_depth": 7, "max_features": 0.1, "min_samples_leaf": 10, "min_samples_split": 100},
}

RF_REG_GRID = {"n_estimators": [500], "max_depth": [3, 4, 5, 7],
               "min_samples_leaf": [50, 100, 200, 300, 500],
               "min_samples_split": [50], "max_features": [0.05, 0.1, 0.2]}


def make_xgb_or_hgb(seed):
    try:
        from xgboost import XGBClassifier
        est = XGBClassifier(subsample=0.8, reg_lambda=10, eval_metric="logloss",
                            random_state=seed, n_jobs=1, tree_method="hist")
        grid = {"max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05], "n_estimators": [200, 400],
                "min_child_weight": [10, 30], "colsample_bytree": [0.3, 0.6]}
        return "xgboost", est, grid
    except ImportError:
        est = HistGradientBoostingClassifier(random_state=seed)
        grid = {"max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05], "max_iter": [200, 400],
                "min_samples_leaf": [50, 100, 200], "l2_regularization": [1.0, 10.0]}
        return "hist_gradient_boosting (xgboost not installed)", est, grid


def lr_pipe(penalty, seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty=penalty, solver="saga", max_iter=3000,
                                            random_state=seed, **({"l1_ratio": 0.5} if penalty == "elasticnet" else {})))


def tune(est, grid, sets, feats, y, groups, n_tune, cv, n_jobs):
    """Grid search on the first n_tune training sets; CV-AUC averaged over the sets."""
    score_sum, params = None, None
    for t in range(n_tune):
        gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, refit=False)
        gs.fit(sets[t][feats], y, groups=groups)
        sc = np.asarray(gs.cv_results_["mean_test_score"], dtype=float)
        score_sum = sc if score_sum is None else score_sum + sc
        params = gs.cv_results_["params"]
    mean = score_sum / n_tune
    i = int(np.argmax(mean))
    return params[i], float(mean[i])


def evaluate(name, build, best, cv_auc, train_sets, test_sets, feats_for, y_train, y_test, g_test, rng, n_boot):
    p_tr, p_te, aucs_te, aucs_tr, n_feat = [], [], [], [], []
    for m, (tr, te) in enumerate(zip(train_sets, test_sets)):
        feats = feats_for(m)
        mdl = build(best, m)
        mdl.fit(tr[feats], y_train)
        a, b = mdl.predict_proba(tr[feats])[:, 1], mdl.predict_proba(te[feats])[:, 1]
        p_tr.append(a)
        p_te.append(b)
        aucs_tr.append(roc_auc_score(y_train, a))
        aucs_te.append(roc_auc_score(y_test, b))
        n_feat.append(len(feats))
    pe = np.mean(p_te, axis=0)
    draws = pp.make_draws(g_test, n_boot, rng)
    boots = [roc_auc_score(y_test[i], pe[i]) for i in draws if 0 < y_test[i].sum() < len(i)]
    row = {"model": name, "n_features": float(np.mean(n_feat)), "cv_auc_tuning": cv_auc,
           "train_auc_mean": float(np.mean(aucs_tr)),
           "test_auc_mean_over_sets": float(np.mean(aucs_te)),
           "test_auc_min": float(np.min(aucs_te)), "test_auc_max": float(np.max(aucs_te)),
           "test_auc_ensemble": float(roc_auc_score(y_test, pe)),
           "ensemble_ci_lo": float(np.percentile(boots, 2.5)), "ensemble_ci_hi": float(np.percentile(boots, 97.5)),
           "params": str(best)}
    print(f"  {name:<16} CV {cv_auc:.3f} | train {row['train_auc_mean']:.3f} | test mean {row['test_auc_mean_over_sets']:.3f} "
          f"| ensemble {row['test_auc_ensemble']:.3f} ({row['ensemble_ci_lo']:.3f}-{row['ensemble_ci_hi']:.3f}) "
          f"| {row['n_features']:.0f} features", flush=True)
    return row


def main():
    args = pp.parse_args()
    t0 = time.time()
    model_no = args.models[0]
    df, y, features, imputation_cols = pp.load_and_prepare(args)
    patients = pp.patient_ids(df)
    is_test, cutoff, _, _ = pp.temporal_split(df["scan_date"], patients, args.test_fraction)

    pop = pd.Series(True, index=df.index)
    if model_no in (3, 4):
        pop = df[D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna()
    train_mask, test_mask = (pop & ~is_test).to_numpy(), (pop & is_test).to_numpy()
    y_train, y_test = y[train_mask].to_numpy(), y[test_mask].to_numpy()
    g_train, g_test = patients[train_mask].to_numpy(), patients[test_mask].to_numpy()
    feats = features[model_no]
    print(f"Model {model_no}: training {train_mask.sum()} scans, test {test_mask.sum()} scans, {len(feats)} features")

    tag = "pop_dd" if (model_no in (3, 4) or args.require_ddimer) else "pop_all"
    sets = pp.impute_sets(df[imputation_cols], train_mask, test_mask, args.n_imputations, args.mice_max_iter,
                          args.seed, args.out_dir / "cache", tag,
                          y_train=y_train if args.outcome_in_train_imputation else None)
    train_sets, test_sets = sets["train"], sets["test"]
    n_tune = max(1, min(args.tuning_sets, len(train_sets)))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed + 100 + model_no)
    nb = 20 if args.quick else 200
    rows = []
    all_feats = lambda m: feats
    nj = args.n_jobs

    print("\nCandidates:")
    # 1. current RF (no tuning: hyperparameters of the main run)
    base = {**BASELINE_PARAMS[model_no], "n_estimators": 500}
    rf = lambda p, m: RandomForestClassifier(**p, random_state=args.seed + m, n_jobs=nj)
    rows.append(evaluate("rf_current", rf, base, np.nan, train_sets, test_sets, all_feats, y_train, y_test, g_test, rng, nb))

    # 2. regularised RF
    with Heartbeat("rf_regularised grid"):
        best, cva = tune(RandomForestClassifier(random_state=args.seed, n_jobs=1),
                         RF_REG_GRID, train_sets, feats, y_train, g_train, n_tune, cv, nj)
    rows.append(evaluate("rf_regularised", rf, best, cva, train_sets, test_sets, all_feats, y_train, y_test, g_test, rng, nb))

    # 3. elastic net and 4. LASSO (standardised, saga)
    en_grid = {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
               "logisticregression__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}
    with Heartbeat("elastic net grid"):
        best_en, cva = tune(lr_pipe("elasticnet", args.seed), en_grid, train_sets, feats, y_train, g_train, n_tune, cv, nj)

    def build_en(p, m):
        return lr_pipe("elasticnet", args.seed).set_params(**p)
    rows.append(evaluate("elastic_net", build_en, best_en, cva, train_sets, test_sets, all_feats, y_train, y_test, g_test, rng, nb))

    la_grid = {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}
    with Heartbeat("lasso grid"):
        best_la, cva = tune(lr_pipe("l1", args.seed), la_grid, train_sets, feats, y_train, g_train, n_tune, cv, nj)

    def build_la(p, m):
        return lr_pipe("l1", args.seed).set_params(**p)
    rows.append(evaluate("lasso", build_la, best_la, cva, train_sets, test_sets, all_feats, y_train, y_test, g_test, rng, nb))

    # 5. XGBoost
    name, est, grid = make_xgb_or_hgb(args.seed)
    with Heartbeat("boosting grid"):
        best_gb, cva = tune(est, grid, train_sets, feats, y_train, g_train, n_tune, cv, nj)
    rows.append(evaluate(name.split(" ")[0], lambda p, m: est.__class__(**{**est.get_params(), **p}),
                         best_gb, cva, train_sets, test_sets, all_feats, y_train, y_test, g_test, rng, nb))

    # 6. LASSO selects variables (within each training set), RF on the selected variables
    selected = {}

    def sel_feats(m):
        if m not in selected:
            lr = lr_pipe("l1", args.seed).set_params(**best_la)
            lr.fit(train_sets[m][feats], y_train)
            coef = lr[-1].coef_.ravel()
            keep = [f for f, c in zip(feats, coef) if abs(c) > 1e-8]
            selected[m] = keep if len(keep) >= 3 else feats
        return selected[m]
    rows.append(evaluate("rf_on_lasso", rf, base, np.nan, train_sets, test_sets, sel_feats, y_train, y_test, g_test, rng, nb))

    out = pd.DataFrame(rows)
    path = args.out_dir / f"try_models_m{model_no}.csv"
    out.to_csv(path, index=False)
    print(f"\nWritten: {path}   ({(time.time() - t0) / 60:.1f} min)")
    print("Paste the table above here (aggregates only).")


if __name__ == "__main__":
    main()
