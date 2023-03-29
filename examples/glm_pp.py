# %%
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dripp.experiments.run_multiple_em_on_cdl import \
    run_cdl_sample, run_cdl_somato
from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel


N_JOBS = 40  # number of jobs to run in parallel. To adjust based on machine

data_source = 'sample'
Delta = 20e-3  # temporal bins
tasks = [1, 2]

# CDL parameters
cdl_params = {
    'n_atoms': 40,
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'reg': 0.1,
    'n_jobs': 5,
    'n_splits': 10
}

# get CDL results
if data_source == 'sample':
    dict_global = run_cdl_sample(**cdl_params)
elif data_source == 'somato':
    dict_global = run_cdl_somato(**cdl_params)

dict_pair_up = dict_global['dict_pair_up']
T = dict_pair_up['T']
events_timestamps = dict_pair_up['events_timestamps']
acti = np.array(dict_pair_up['acti_shift'])
z_hat = np.array(dict_global['dict_cdl_fit_res']['z_hat'])

# %%
if isinstance(tasks, list):
    tt = np.r_[events_timestamps[tasks[0]]]
    for i in tasks[1:]:
        tt = np.r_[tt, events_timestamps[i]]
elif isinstance(tasks, int):
    tt = events_timestamps[tasks]

tt_sparse = np.zeros(int(T*150))
tt_sparse[tt] = 1


def binarized(process, Delta, sfreq):
    """

    Delta : time bin, in seconds
    """

    bin_tt = np.round(Delta * sfreq)  # length in tt of one bin
    l = np.round(len(process) / bin_tt)
    process_binarized = np.array([])

    return process_binarized

# %%


# %%
