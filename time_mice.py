"""
Timing test for MICE on the Model 3/4 population (scans with a D-dimer).
Times one IterativeImputer pass (max_iter=1) on the training set with 1 BLAS
thread and with the default threads. Run on MyDRE via run_time_mice.py.
"""
import time

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from threadpoolctl import threadpool_info, threadpool_limits

import pe_pipeline as pp


def timed(label, frame, limit):
    imputer = IterativeImputer(sample_posterior=True, random_state=0, max_iter=1,
                               initial_strategy="median")
    t0 = time.time()
    if limit:
        with threadpool_limits(limits=limit):
            imputer.fit_transform(frame)
    else:
        imputer.fit_transform(frame)
    print(f"{label}: {time.time() - t0:.1f} s", flush=True)


def main():
    args = pp.parse_args()
    args.models = [3]
    df, y, features, imputation_cols = pp.load_and_prepare(args)
    is_test, *_ = pp.temporal_split(df["scan_date"], pp.patient_ids(df), args.test_fraction)
    pop = df[pp.D_DIMER_VALUE_COL].notna() & df["D_dimer_assay_enc"].notna()
    frame = df.loc[pop & ~is_test, imputation_cols]
    frame = frame[[c for c in frame.columns if frame[c].notna().any()]]
    print(f"Training frame: {frame.shape[0]} rows x {frame.shape[1]} columns", flush=True)
    print("Missing share per column (highest 8):\n"
          + frame.isna().mean().sort_values(ascending=False).head(8).round(3).to_string(), flush=True)
    print("Max abs value per column (highest 5):\n"
          + frame.abs().max().sort_values(ascending=False).head(5).to_string(), flush=True)
    print("BLAS/OpenMP pools:", [(d["internal_api"], d["num_threads"]) for d in threadpool_info()], flush=True)
    timed("1 pass, 1 thread", frame, 1)
    timed("1 pass, default threads", frame, None)


if __name__ == "__main__":
    main()
