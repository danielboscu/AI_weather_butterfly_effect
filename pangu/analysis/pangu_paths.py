import os
import numpy as np

# Root for run output (data/).
pangu_scratch = os.environ["pangu_scratch"]

data_root = f"{pangu_scratch}/data"

def run_dir(precision, mode, source, noise, scale, step=None):
    p = f"{data_root}/{precision}/{mode}/{source}/{noise}/{scale}"
    if step is None:
        return p
    return f"{p}/{step}h"

diag_vars = ("u300", "v300", "u850", "z500")
def load_diagnostics(precision, mode, source, noise, scale, step=None):
    d = run_dir(precision, mode, source, noise, scale, step=step) + "/diag"
    if not os.path.exists(f"{d}/.distilled_ok"):
        raise FileNotFoundError(f"{d} is missing or not verified; run distill_run.py first")
    arrays = {}
    for v in diag_vars:
        arrays[v] = np.load(f"{d}/{v}.npy", mmap_mode="r")
    return arrays, np.load(f"{d}/hours.npy")
