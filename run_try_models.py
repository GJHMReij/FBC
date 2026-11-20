"""
Runner for try_models.py on MyDRE: compares the current Random Forest with a regularised RF,
elastic net, LASSO, XGBoost and RF-on-LASSO-selected-variables for ONE model (set MODEL below),
on the same temporal split and the same cached MICE sets (OUT_DIR = results_temporal_outcome_imp).
Prints an aggregate table only. QUICK = True is a smoke test; set False for the real run.
Edit the settings, save, press F5 in Spyder.
"""
import subprocess
import sys

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
QUICK = True
MODEL = 1
OUT_DIR = FOLDER + "\\results_temporal_outcome_imp"

subprocess.run(["git", "pull"], cwd=FOLDER)
cmd = [sys.executable, "-u", "try_models.py", "--models", str(MODEL),
       "--input-csv", INPUT_CSV, "--outcome-correction", OUTCOME_CORRECTION,
       "--outcome-in-train-imputation", "--tuning-sets", "3", "--out-dir", OUT_DIR]
if QUICK:
    cmd.append("--quick")
p = subprocess.Popen(cmd, cwd=FOLDER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="")
p.wait()
