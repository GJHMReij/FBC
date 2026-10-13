"""
Runs Model 1, 2 and 3 one after the other on the real MyDRE cohort (git pull first).
Each model is its own run (Model 3 builds its own split, after excluding scans
without D-dimer). If one model fails, the next one still runs.

QUICK = True does a fast smoke test (2 imputations, tiny grid, 20 bootstrap
draws). Set QUICK = False for the real, full runs (several hours in total).
Edit the settings below, save, and press F5 (Run file) in Spyder.
"""
import subprocess
import sys
from datetime import datetime

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
QUICK = True
MODELS = [1, 2, 3]

STAMP = f"{datetime.now():%Y%m%d_%H%M%S}"


def run_and_stream(cmd, cwd, log_path=None):
    """Run cmd and print its output line by line; also write it to log_path."""
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
    return process.returncode


run_and_stream(["git", "pull"], cwd=FOLDER)

failed = []
for model_no in MODELS:
    log_path = f"{FOLDER}\\model{model_no}_pipeline_log_{STAMP}.txt"
    print(f"\n{'#' * 70}\n# MODEL {model_no}  ({'QUICK' if QUICK else 'FULL'} run)\n{'#' * 70}")
    cmd = [
        sys.executable, "-u", "pe_pipeline.py",
        "--models", str(model_no),
        "--input-csv", INPUT_CSV,
        "--outcome-correction", OUTCOME_CORRECTION,
    ]
    if QUICK:
        cmd.append("--quick")
    code = run_and_stream(cmd, cwd=FOLDER, log_path=log_path)
    if code != 0:
        failed.append(model_no)
        print(f"\nMODEL {model_no} FAILED (exit code {code}); continuing with the next model")

print("\nDone. Reports: results_temporal\\model{n}_report.docx")
if failed:
    print(f"FAILED models: {failed} -- see the logs above / in {FOLDER}")
