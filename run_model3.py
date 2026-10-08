"""
Convenience runner for Model 3 on the real MyDRE cohort -- pulls the latest
code, then runs the 80/20 temporal-split pipeline (pe_pipeline.py) for Model 3.
Edit the settings below, save, and press F5 (Run file) in Spyder.

QUICK = True does a fast smoke test (2 imputations, tiny grid, 20 bootstrap
draws) -- use it first to check that everything runs and that the printed
split numbers look right. Set QUICK = False for the real, full run.
"""
import subprocess
import sys
from datetime import datetime

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
QUICK = True

LOG_PATH = f"{FOLDER}\\model3_pipeline_log_{datetime.now():%Y%m%d_%H%M%S}.txt"


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
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {process.returncode}): {' '.join(cmd)}")


run_and_stream(["git", "pull"], cwd=FOLDER)

cmd = [
    sys.executable, "-u", "pe_pipeline.py",
    "--models", "3",
    "--input-csv", INPUT_CSV,
    "--outcome-correction", OUTCOME_CORRECTION,
]
if QUICK:
    cmd.append("--quick")
run_and_stream(cmd, cwd=FOLDER, log_path=LOG_PATH)

print(f"\nFull run transcript saved to: {LOG_PATH}")
