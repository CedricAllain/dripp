# %%

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel

# subject_id = "CC110037"  # 18.75
# subject_id = "CC620264"  # 76.33
# subject_id = "CC723395"  # 86.08
subject_id = "CC320428"  # 45.58 Male


N_JOBS = 10  # number of jobs to run in parallel. To adjust based on machine

# CDL parameters
cdl_params = {
    'subject_id': subject_id,
    'n_atoms': 30,
    'sfreq': 150.,
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
dict_global, df_res = run_multiple_em_on_cdl(
    data_source='camcan', cdl_params=cdl_params,  # CDL
    shift_acti=shift_acti, atom_to_filter='all', threshold=threshold,
    list_atoms=list(range(cdl_params['n_atoms'])),
    list_tasks=([1, 2, 3], [4]),  # audiovis and button stimuli, in two drivers
    n_driver=2,
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',  # EM
    n_jobs=N_JOBS)
