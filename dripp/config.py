from pathlib import Path

# path to directory where to save multiple results from dripp/example
# and dripp/benchmarks
SAVE_RESULTS_PATH = Path('./dripp_results')
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')
if not CACHEDIR.exists():
    CACHEDIR.mkdir(parents=True)

N_JOBS = 10  # number of jobs to run in parallel. To adjust based on machine
