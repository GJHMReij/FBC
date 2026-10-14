# Handoff: ML analysis FBC / PE project

Researcher: Max Reijers (PhD, Amsterdam UMC). Speaks Dutch; manuscript is in English. Wants step-by-step "how to run" instructions every time. Real patient data lives only on the secure MyDRE Windows server (never on the Mac). Code is developed here against the dummy CSV, pushed to GitHub (`https://github.com/GJHMReij/FBC`, branch `main`), then pulled and run on MyDRE in Spyder (folder `C:\Users\Max.Reijers\Documents\FBC_pe_models`). On MyDRE: `%cd <folder>` then `!git pull` in the Spyder console (old git: no `-C`).

## Goal
Predict pulmonary embolism (PE) from CBC/DIFF blood parameters (+ D-dimer) to rule out PE and save CTPA scans. Three Random Forest models:
- Model 1: age, sex, creatinine, CBC
- Model 2: Model 1 + DIFF parameters
- Model 3: Model 2 + D-dimer level and assay (scans without D-dimer excluded before the split, not imputed)

## Current analysis plan (implemented in `pe_pipeline.py`)
1. Exclusions: outpatient scans, non-assessable CTPA, missing date/sex/age. Model 3 additionally excludes scans without `d_dimeer48H` BEFORE the split.
2. Temporal split: hold-out test set = exactly 20% of scans, built per patient from the most recent first-scan dates backwards; patients with scans in both periods go entirely to training.
3. MICE (10 imputation sets, separately for training and test; outcome not in the imputation model).
4. Grid search (5-fold patient-grouped CV, AUC) on training imputation set 1 (grid per Kasia, 324 combos).
5. One RF per imputation set; the mean probability of the 10 RFs is the model. Logistic recalibration fitted on training cross-validated predictions (`--no-recalibration` to skip).
6. Thresholds for sensitivity 97/98/99% determined on 5-fold cross-validated training predictions, applied unchanged to the test set (Table 5 / "from training"; table "sensitivity fixed on test" is the optimistic reference). Rubin's rules + patient-level bootstrap CIs. Brier/AUC are threshold-independent.
7. Outputs in `results_temporal/` (per model: `_report.docx`, ROC and calibration PNGs, per-1000-scans tables, `_predictions.csv`, `_ensemble.joblib`, ...).

## Runners (on MyDRE, Spyder, F5)
- `run_model1.py`, `run_model2.py`, `run_model3.py`, `run_all_models.py` (QUICK = True is a smoke test; QUICK = False is the real run, several hours).
- `flowchart_numbers.py` prints/writes the selection counts; `figures/make_flowchart.py` writes `figures/flowchart_selection.svg` (green rounded-box style matching the user's slides).

## Findings so far (quick runs, not final)
- Model 1 test AUC about 0.63 (weak; Brier about 0.14 = baseline), well calibrated after removing `class_weight="balanced"`.
- Model 3 test AUC about 0.76; raw calibration slope about 2.0, about 1.06 after recalibration.
- Real-data counts: Models 1/2: 7,658 scans (training 6,126 / test 1,532, PE 18.1% / 17.6%); Model 3: 4,748 scans (training 3,798 / test 950, PE 16.9% / 17.3%).
- Class balancing removed because it inflated probabilities; the calibration curve cannot span 0-1 with weak predictors (explained to the user).

## Open items
- Run the quick versions of all three models with the latest code (cross-validated thresholds + recalibration have never been executed on MyDRE), check Table 5, then do the full runs (`QUICK = False`).
- Flowchart: top number 8,202 is derived (7,690 + 512) and the 7,690 / 6 non-assessable split is inferred from 568 outpatient (R) vs 566 (script); verify in R. 25 scans lack a valid date (Models 1/2): check why.
- Compare Model 3 with D-dimer alone (conventional and age-adjusted cut-off) for the manuscript.
- `fill_manuscript.py` and `combine_roc_curves.py` still read the old file locations; update to `results_temporal/`.
- External validation (UCLH/Barts) and per-hospital analyses are optional; user said per-hospital reporting is not needed in the plan.
