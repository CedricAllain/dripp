from pathlib import Path

# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')
if not CACHEDIR.exists():
    CACHEDIR.mkdir(parents=True)

# path to directory where to save multiple results from dripp/example
# and dripp/benchmarks
SAVE_RESULTS_PATH = Path('./dripp_results')
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)
