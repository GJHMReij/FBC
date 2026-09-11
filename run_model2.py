"""
Convenience runner for Model 2 on the real MyDRE cohort -- pulls the latest
code, then runs the model with the outcome correction applied. Edit the
paths/settings below as needed, save, and press Run in Spyder.
"""
import subprocess
import sys

FOLDER = r"C:\Users\Max.Reijers\Documents\FBC_pe_models"
INPUT_CSV = r"C:\Users\Max.Reijers\Desktop\models august\CohortMLgeslacht.csv"
OUTCOME_CORRECTION = r"C:\Users\Max.Reijers\Desktop\models august\df_met_script8000.xlsx"
N_BOOTSTRAP = "50"  # TODO: set back to 500 for the final, definitive run
N_IMPUTATIONS = "10"


def run_and_stream(cmd, cwd):
    """Run cmd and print its output line-by-line as it happens (via
    Python's own print, which Spyder's console does capture and display --
    unlike a plain subprocess.run(), whose inherited stdout can go
    nowhere visible when Spyder itself has no attached console window)."""
    print(f"$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {process.returncode}): {' '.join(cmd)}")


run_and_stream(["git", "pull"], cwd=FOLDER)

run_and_stream([
    # -u: force the child Python process to run unbuffered, so its print()
    # output is flushed immediately instead of sitting in an internal
    # buffer (the default when stdout isn't a real terminal, as here) --
    # without this, no output appears until the buffer fills or the
    # process exits, even though it's running fine the whole time.
    sys.executable, "-u", "pe_model2_cbc_diff_rf.py",
    "--input-csv", INPUT_CSV,
    "--outcome-correction", OUTCOME_CORRECTION,
    "--n-bootstrap", N_BOOTSTRAP,
    "--n-imputations", N_IMPUTATIONS,
], cwd=FOLDER)
