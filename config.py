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

# paths to CamCAN files for Inria Saclay users
DATA_DIR = Path("/storage/store/data/")
SSS_CAL = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
BIDS_root = DATA_DIR / "camcan/BIDSsep/smt/"
PARTICIPANTS_FILE = BIDS_root / "participants.tsv"
