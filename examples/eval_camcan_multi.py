# %%

import numpy as np
from pathlib import Path

import mne
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from mne_bids import BIDSPath, read_raw_bids

from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
# from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH
from dripp.experiments.utils_plot import plot_cdl_atoms
# from dripp.trunc_norm_kernel.model import TruncNormKernel

DATA_SOURCE = 'camcan'

# subject_id = "CC110037"  # 18.75
subject_id = "CC620264"  # 76.33 Female
# subject_id = "CC723395"  # 86.08
# subject_id = "CC320428"  # 45.58 Male
# %%

N_JOBS = 10  # number of jobs to run in parallel. To adjust based on machine

atom_duration = 0.5
sfreq = 150.

# CDL parameters
cdl_params = {
    'subject_id': subject_id,
    'use_greedy': True,
    'n_atoms': 20,
    'n_times_atom': int(np.round(atom_duration * sfreq)),
    'sfreq': sfreq,
    'n_iter': 100,
    'eps': 1e-5,
    'reg': 0.2,
    'tol_z': 1e-3,
    'n_jobs': 5,
    'n_splits': 10
}
# run CDL and EM
lower, upper = 0, 900e-3
shift_acti = True
threshold = 0
n_iter = 200
dict_global, df_res_dripp = run_multiple_em_on_cdl(
    data_source=DATA_SOURCE, cdl_params=cdl_params,  # CDL
    shift_acti=shift_acti, atom_to_filter='all', threshold=threshold,
    list_atoms=list(range(cdl_params['n_atoms'])),
    # , [4]),  # audiovis and button stimuli, in two drivers
    list_tasks=([1, 2, 3, 5], [1, 2, 3, 6]),
    n_driver=2,
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',
    n_jobs=N_JOBS)

# %%

# plotted_tasks = {'audivis': [1, 2, 3],
#                  'button': [4]}

plotted_tasks = {'audivis_catch0': [1, 2, 3, 5],
                 'audivis_catch1': [1, 2, 3, 6]}
n_top_atoms = 5

# get raw info
# %% Read raw data from BIDS file
DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
bp = BIDSPath(
    root=BIDS_ROOT,
    subject=subject_id,
    task="smt",
    datatype="meg",
    extension=".fif",
    session="smt",
)
raw = read_raw_bids(bp)
raw.pick_types(meg='grad', eeg=False, eog=False, stim=True)
info = raw.copy().pick_types(meg=True).info

# %%

# ================================================================
# PLOT ALL EXTRACTED ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ================================================================

if cdl_params['use_greedy']:
    method = "greedy"
else:
    method = "batch"

fig_name = DATA_SOURCE + '_' + subject_id + '_' + method + '_'
if n_top_atoms is not None:
    fig_name += str(n_top_atoms) + '_top_atoms.pdf'
else:
    fig_name += 'all_atoms.pdf'

plot_cdl_atoms(dict_global, cdl_params, info,
               n_top_atoms=n_top_atoms, plot_intensity=True,
               df_res_dripp=df_res_dripp, plotted_tasks=plotted_tasks,
               save_fig=True, path_fig=SAVE_RESULTS_PATH / fig_name)

# %%
