"""
Timing test for the MICE step of Model 3/4 on MyDRE: pulls the latest code and
times ONE imputation pass (max_iter=1) of the D-dimer population, with 1 BLAS
thread and with the default threads. Takes a few minutes at most; no results
are written. QUICK is not used.
"""
import subprocess
import sys
from datetime import datetime

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\df_met_script8000.xlsx"
QUICK = False

LOG_PATH = f"{FOLDER}\\time_mice_log_{datetime.now():%Y%m%d_%H%M%S}.txt"


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
    sys.executable, "-u", "time_mice.py",
    "--input-csv", INPUT_CSV,
    "--outcome-correction", OUTCOME_CORRECTION,
]
if QUICK:
    cmd.append("--quick")
run_and_stream(cmd, cwd=FOLDER, log_path=LOG_PATH)

print(f"\nFull run transcript saved to: {LOG_PATH}")
