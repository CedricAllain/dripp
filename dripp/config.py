import os

from pathlib import Path

# path to directory where to save multiple results from dripp/example
# and dripp/benchmarks
SAVE_RESULTS_PATH = Path(os.getenv('DRIPP_RESULTS', './dripp_results'))
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

# Define the cache directory for joblib.Memory
CACHEDIR = Path(os.getenv('DRIPP_RESULTS_CACHE', './__cache__'))
if not CACHEDIR.exists():
    CACHEDIR.mkdir(parents=True)
