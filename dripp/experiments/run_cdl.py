# %%
"""
Run Convolutional Dictionary Learning on mne.sample or mne.somato dataset
"""

import os
import mne
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Memory

from alphacsc import GreedyCDL, BatchCDL
from alphacsc.utils.signal import split_signal

from mne_bids import BIDSPath, read_raw_bids

from dripp.cdl import utils
from dripp.config import CACHEDIR, SAVE_RESULTS_PATH


memory = Memory(CACHEDIR, verbose=0)

# for Cam-CAN dataset
DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

# %%


@memory.cache(ignore=['n_jobs'])
def _run_cdl_data(sfreq=150., n_atoms=40, n_times_atom=None, reg=0.1,
                  n_iter=100, eps=1e-4, tol_z=1e-3, use_greedy=True,
                  n_jobs=5, n_splits=10,
                  data_source='sample', subject_id='CC620264',
                  save_results=False):
    """Run a Greedy Convolutional Dictionary Learning on mne.[data_source]
    dataset.

    Parameters
    ----------
    data_source : str, 'sample' | 'camcan' | 'somato'
        Data source name. Defaults to 'sample'

    subject_id : str
        For Cam-CAN dataset, the subject id to run the CSC on. Defaults to
        'CC620264', a 76.33 year old woman.

    subject : str
        Subject label for camcan dataset, e.g., subject = 'CC110033'. Defaults
        to 'sample'.

    kind : 'passive' | 'rest' | 'task'
        Only for camcan dataset, kind of experiment done on the subject.
        Defaults to 'passive'.

    sfreq : double
        Sampling frequency. The signal will be resampled to match this.
        Defaults to 150.

    n_atoms : int
        Number of atoms to learn. Defaults to 40.

    n_times_atoms : int | None
        The support of the atom (in timestamps). If None, set to sfreq.
        Defaults to None.

    reg : double
        Regularization parameter which control sparsity. Defaults to 0.1.

    n_iter : int
        Number of iteration for the alternate minimization. Defaults to 100.

    eps : float
        Convergence threshold. Defaults to 1e-4.

    use_greedy: bool
        If True, use GreedyCDL, if false, use BatchCDL. Defaults to True.

    n_jobs : int
        Number of processors for parallel computing. Defaults to 5.

    n_splits : int
        Number of splits the raw signal is decomposed into. The number of
        splits should actually be the smallest possible to avoid
        introducing border artifacts in the learned atoms and it should be no
        much larger than n_jobs.
        A good value is n_splits = n_jobs, or n_splits set to be a small
        multiple of n_jobs. Defaults to 10.

    Returns
    -------
    dict_global : dict of dict
        Global dictionary with keys as follow.

        'dict_cdl_params' : dict
            Value of GreedyCDL's parameters.

        'dict_other_params' : dict
            Value of all other parameters, such as data source, sfreq, etc.

        'dict_cdl_fit_res' : dict of numpy.array
            Results of the cdl.fit(), with u_hat_, v_hat_ and z_hat.

        'dict_pair_up' : dict
            Pre-process of results that serve as input in a EM algorithm.
    """
    print("Run CDL model")

    if n_times_atom is None:
        n_times_atom = int(round(sfreq * 1.0))

    # recompute n_jobs and n_splits so it is optimal,
    # i.e., n_splits is a multiple of n_jobs
    k = n_splits // n_jobs
    n_jobs = min(n_jobs, os.cpu_count())
    n_splits = min(n_splits, n_jobs * k)

    # get dataset utils
    data_utils = utils.get_data_utils(data_source=data_source, verbose=True)

    # Load data and preprocessing
    raw, events, event_id, _ = utils.raw_preprocessing(
        data_source=data_source, subject_id=subject_id, sfreq=sfreq)

    X = raw.get_data(picks=['meg'])
    X_split = split_signal(X, n_splits=n_splits, apply_window=True)

    # Define Greedy Convolutional Dictionary Learning model
    cdl_params = {
        # Shape of the dictionary
        'n_atoms': n_atoms,
        'n_times_atom': n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        'rank1': True,
        'uv_constraint': 'separate',
        # apply a temporal window reparametrization
        'window': True,
        # at the end, refit the activations with fixed support
        # and no reg to unbias
        'unbiased_z_hat': True,
        # Initialize the dictionary with random chunk from the data
        'D_init': 'chunk',
        # rescale the regularization parameter to be a percentage of lambda_max
        'lmbd_max': "scaled",  # original value: "scaled"
        'reg': reg,
        # Number of iteration for the alternate minimization and cvg threshold
        'n_iter': n_iter,  # original value: 100
        'eps': eps,  # original value: 1e-4
        # solver for the z-step
        'solver_z': "lgcd",
        'solver_z_kwargs': {'tol': tol_z,  # stopping criteria
                            'max_iter': 100000},
        # solver for the d-step
        'solver_d': 'alternate_adaptive',
        'solver_d_kwargs': {'max_iter': 300},  # original value: 300
        # sort atoms by explained variances
        'sort_atoms': True,
        # Technical parameters
        'verbose': 1,
        'random_state': 0,
        'n_jobs': n_jobs
    }

    if use_greedy:
        cdl = GreedyCDL(**cdl_params)
    else:
        cdl = BatchCDL(**cdl_params)

    # Fit the model
    cdl.fit(X_split)
    z_hat = cdl.transform(X[None, :])  # compute atoms activation intensities

    # Duration of the experiment, in seconds
    T = z_hat.shape[2] / sfreq

    # Determine events timestamps and activation vectors
    events_timestamps = utils.get_events_timestamps(events=events,
                                                    sfreq=sfreq,
                                                    event_id=event_id)
    acti_shift = utils.get_activation(model=cdl,
                                      z_hat=z_hat.copy(),
                                      shift=True)
    acti_not_shift = z_hat[0].copy()

    # Construct parameters dictionaries and save the model
    dict_cdl_params = utils.get_dict_cdl_params(cdl)

    dict_other_params = {'data_source': data_source,
                         'sfreq': sfreq,
                         'n_splits': n_splits,
                         'event_id': event_id}
    # if data_source == 'camcan':
    #     dict_other_params = dict(dict_other_params,
    #                              **{'age': age,
    #                                 'sex': sex,
    #                                 'subject': subject_id})
    # elif data_source in ['sample', 'somato']:
    #     dict_other_params = dict(dict_other_params,
    #                              **{'file_name': file_name,
    #                                 'stim_channel': stim_channel})

    dict_cdl_fit_res = {'u_hat_': cdl.u_hat_.tolist(),
                        'v_hat_': cdl.v_hat_.tolist(),
                        'z_hat': z_hat.tolist()}

    dict_pair_up = {'T': T,
                    'events': events,
                    'events_timestamps': events_timestamps,
                    'acti_shift': acti_shift,
                    'acti_not_shift': acti_not_shift}

    dict_global = {'dict_cdl_params': dict_cdl_params,
                   'dict_other_params': dict_other_params,
                   'dict_cdl_fit_res': dict_cdl_fit_res,
                   'dict_pair_up': dict_pair_up}

    if save_results:
        # save results in JSON file
        json_file_path = SAVE_RESULTS_PATH / ('cdl_' + data_source + '.json')
        with open(json_file_path, 'w') as fp:
            json.dump(dict_global, fp, sort_keys=True, indent=4,
                      cls=utils.NumpyEncoder)

    return dict_global


def run_cdl_sample(sfreq=150., n_atoms=40, n_times_atom=None, reg=0.1,
                   n_iter=100, eps=1e-4, n_jobs=5, n_splits=10):
    """Run Convolutional Dictionary Learning on mne.sample."""
    return _run_cdl_data(sfreq=sfreq, n_atoms=n_atoms,
                         n_times_atom=n_times_atom, reg=reg,
                         n_iter=n_iter, eps=eps, n_jobs=n_jobs,
                         n_splits=n_splits,
                         data_source='sample')


def run_cdl_somato(sfreq=150., n_atoms=25, n_times_atom=None, reg=0.2,
                   n_iter=100, eps=1e-4, use_greedy=False, n_jobs=5,
                   n_splits=10):
    """Run Convolutional Dictionary Learning on mne.somato."""
    return _run_cdl_data(sfreq=sfreq, n_atoms=n_atoms,
                         n_times_atom=n_times_atom, reg=reg,
                         n_iter=n_iter, eps=eps, n_jobs=n_jobs,
                         n_splits=n_splits,
                         data_source='somato')


def run_cdl_camcan(subject_id="CC320428", sfreq=150., n_atoms=30,
                   n_times_atom=int(np.round(0.7*150.)), reg=0.2, n_iter=100,
                   eps=1e-5, tol_z=1e-3, use_greedy=False, n_jobs=5,
                   n_splits=10):
    """Run Convolutional Dictionary Learning on Cam-CAN dataset.

    Parameters
    ----------
    subject_id : str
        For Cam-CAN dataset, the subject id to run the CSC on. Defaults to
        'CC620264', a 76.33 year old woman.

    """
    return _run_cdl_data(sfreq=sfreq, n_atoms=n_atoms,
                         n_times_atom=n_times_atom, reg=reg,
                         n_iter=n_iter, eps=eps, tol_z=tol_z,
                         use_greedy=use_greedy,
                         n_jobs=n_jobs, n_splits=n_splits,
                         data_source='camcan', subject_id=subject_id)
