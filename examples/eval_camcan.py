# %%

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

import mne
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from joblib import Parallel, delayed

from mne_bids import BIDSPath, read_raw_bids

from dripp.experiments.run_cdl import \
    BIDS_ROOT, PARTICIPANTS_FILE, run_cdl_camcan
from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl

# from dripp.cdl import utils
from dripp.cdl import utils
from dripp.config import N_JOBS, SAVE_RESULTS_PATH
from dripp.experiments.utils_plot import plot_cdl_atoms
# from dripp.trunc_norm_kernel.model import TruncNormKernel

DATA_SOURCE = 'camcan'

SAVE_RESULTS_PATH /= 'results_camcan'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

subject_id = "CC110037"  # 18.75
# subject_id = "CC620264"  # 76.33 Female
# subject_id = "CC723395"  # 86.08
# subject_id = "CC320428"  # 45.58 Male
utils.get_info_camcan(subject_id)

atom_duration = 0.5
sfreq = 150.
# %%


def eval_camcan(subject_id):
    # %%
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
        n_drivers=2,
        lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',
        alpha_pos=True,
        n_jobs=N_JOBS)

    # %%

    # plotted_tasks = {'audivis': [1, 2, 3],
    #                  'button': [4]}

    plotted_tasks = {'audivis_catch0': [1, 2, 3, 5],
                     'audivis_catch1': [1, 2, 3, 6]
                     #  'button': [4]
                     }
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

    for n_top_atoms in [5, None]:
        fig_name = DATA_SOURCE + '_' + subject_id + '_'  # + method + '_'
        if n_top_atoms is not None:
            fig_name += str(n_top_atoms) + '_top_atoms.pdf'
        else:
            fig_name += 'all_atoms.pdf'

        plot_cdl_atoms(dict_global, cdl_params, info,
                       n_top_atoms=n_top_atoms, plot_intensity=True,
                       df_res_dripp=df_res_dripp, plotted_tasks=plotted_tasks,
                       save_fig=True, path_fig=SAVE_RESULTS_PATH / fig_name)


# %%

if __name__ == '__main__':
    n_jobs = 1
    # subject_ids = ["CC110037", "CC620264", "CC723395",
    #                "CC420462", "CC520597", "CC510639", "CC121111"]
    subject_ids = ["CC520597", "CC620264", "CC723395"]
    if n_jobs > 1:
        _ = Parallel(n_jobs=min(len(subject_ids), n_jobs), verbose=1)(
            delayed(eval_camcan)(this_subject_id)
            for this_subject_id in subject_ids)
    else:
        for this_subject_id in subject_ids:
            utils.get_info_camcan(this_subject_id)
            eval_camcan(this_subject_id)

# Subject ID: CC620264, 76.33 year old FEMALE
# Counter events:  Counter({4: 129, 1: 40, 2: 40, 3: 40, 6: 4, 5: 4})
# Experiment duration,  541.5066666666667
# Subject ID: CC723395, 86.08 year old FEMALE
# Counter events:  Counter({4: 132, 1: 40, 2: 40, 3: 40, 6: 4, 5: 4})
# Experiment duration,  544.5066666666667
# Subject ID: CC520597, 64.25 year old MALE
# Counter events:  Counter({4: 131, 1: 40, 2: 40, 3: 40, 6: 4, 5: 4})
# Experiment duration,  543.5066666666667
