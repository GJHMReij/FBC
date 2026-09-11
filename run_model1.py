"""
Convenience runner for Model 1 on the real MyDRE cohort -- pulls the latest
code, then runs the model with the outcome correction applied. Edit the
paths/settings below as needed, save, and press F5 (Run file) in Spyder.
"""
import subprocess
import sys

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\models august\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\models august\df_met_script8000.xlsx"
N_BOOTSTRAP = "500"

subprocess.run(["git", "pull"], cwd=FOLDER, check=True)

subprocess.run([
    sys.executable, "pe_model1_cbc_rf.py",
    "--input-csv", INPUT_CSV,
    "--outcome-correction", OUTCOME_CORRECTION,
    "--n-bootstrap", N_BOOTSTRAP,
], cwd=FOLDER, check=True)
