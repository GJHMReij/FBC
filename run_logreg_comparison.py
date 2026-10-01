"""
Convenience runner for the logistic regression comparison on the real MyDRE
cohort -- pulls the latest code, then runs compare_logreg.py with the
outcome correction applied. Edit the paths/settings below as needed, save,
and press Run in Spyder.
"""
import subprocess
import sys
from datetime import datetime

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
N_IMPUTATIONS = "10"

LOG_PATH = f"{FOLDER}\\logreg_comparison_run_log_{datetime.now():%Y%m%d_%H%M%S}.txt"


def run_and_stream(cmd, cwd, log_path=None):
    header = f"$ {' '.join(cmd)}"
    print(header)
    log_file = open(log_path, "a", encoding="utf-8") if log_path else None
    if log_file:
        log_file.write(header + "\n")
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        if log_file:
            log_file.write(line)
    process.wait()
    if log_file:
        log_file.close()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {process.returncode}): {' '.join(cmd)}")


run_and_stream(["git", "pull"], cwd=FOLDER)

run_and_stream([
    sys.executable, "-u", "compare_logreg.py",
    "--input-csv", INPUT_CSV,
    "--outcome-correction", OUTCOME_CORRECTION,
    "--n-imputations", N_IMPUTATIONS,
], cwd=FOLDER, log_path=LOG_PATH)

print(f"\nFull run transcript saved to: {LOG_PATH}")
