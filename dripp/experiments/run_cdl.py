"""
Run Convolutional Dictionary Learning on mne.sample or mne.somato dataset
"""

import os
import mne
import json
import numpy as np
from joblib import Memory

from alphacsc import GreedyCDL
from alphacsc.utils.signal import split_signal

from dripp.cdl import utils
from dripp.config import CACHEDIR, SAVE_RESULTS_PATH


memory = Memory(CACHEDIR, verbose=0)


@memory.cache(ignore=['n_jobs'])
def _run_cdl_data(sfreq=150., n_atoms=40, n_times_atom=None, reg=0.1,
                  n_iter=100, eps=1e-4, n_jobs=5, n_splits=10,
                  data_source='sample'):
    """Run a Greedy Convolutional Dictionary Learning on mne.[data_source]
    dataset

    Parameters
    ----------
    data_source : str, 'sample' | 'camcan' | 'somato'
        Data source name
        Default is 'sample'

    subject : str
        subject label for camcan dataset, e.g., subject = 'CC110033'
        default is 'sample'

    kind : 'passive' | 'rest' | 'task'
        only for camcan dataset, kind of experiment done on the subject
        default is 'passive'

    sfreq : double
        Sampling frequency. The signal will be resampled to match this.
        Default is 150.

    n_atoms : int
        Number of atoms to learn
        Default is 40

    n_atoms : int
        The support of the atom (in timestamps)
        Default is None (computed after from sfreq)

    reg : double
        Regularization parameter which control sparsity
        Default is 0.1

    n_iter : int
        Number of iteration for the alternate minimization
        Default is 100

    eps : float
        cvg threshold
        Default is 1e-4

    n_jobs : int
        Number of processors for parallel computing
        Default is 5

    n_splits : int
        Number of splits the raw signal is decomposed into
        The number of splits should actually be the smallest possible to avoid
        introducing border artifacts in the learned atoms and it should be no
        much larger than n_jobs.
        A good value is n_splits = n_jobs, or n_splits set to be a small
        multiple of n_jobs.
        Default is 10

    Returns
    -------
    dict_global : dict of dict
        Global dictionary with keys as follow.

        'dict_cdl_params' : dict
            value of GreedyCDL's parameters

        'dict_other_params' : dict
            value of all other parameters, such as data source, sfreq, etc.

        'dict_cdl_fit_res' : dict of numpy.array
            results of the cdl.fit(), with u_hat_, v_hat_ and z_hat

        'dict_pair_up' : dict
            pre-process of results that serve as input in a EM algorithm

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
    print("Loading the data...", end=' ', flush=True)
    file_name = data_utils['file_name']
    raw = mne.io.read_raw_fif(file_name, preload=True, verbose=False)
    raw.pick_types(meg='grad', eeg=False, eog=False, stim=True)
    print('done')

    print("Preprocessing the data...", end=' ', flush=True)
    if data_source == 'sample':
        raw.notch_filter(np.arange(60, 181, 60))
    elif data_source == 'somato':
        raw.notch_filter(np.arange(50, 101, 50))

    raw.filter(l_freq=2, h_freq=None)

    stim_channel = data_utils['stim_channel']
    events = mne.find_events(raw, stim_channel=stim_channel)
    raw, events = raw.resample(
        sfreq, npad='auto', verbose=False, events=events)
    # Set the first sample to 0 in event stim
    events[:, 0] -= raw.first_samp
    print('done')

    X = raw.get_data(picks=['meg'])
    X_split = split_signal(X, n_splits=n_splits, apply_window=True)

    # Define Greedy Convolutional Dictionary Learning model
    cdl = GreedyCDL(
        # Shape of the dictionary
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        rank1=True,
        uv_constraint='separate',
        # apply a temporal window reparametrization
        window=True,
        # at the end, refit the activations with fixed support
        # and no reg to unbias
        unbiased_z_hat=True,
        # Initialize the dictionary with random chunk from the data
        D_init='chunk',
        # rescale the regularization parameter to be a percentage of lambda_max
        lmbd_max="scaled",  # original value: "scaled"
        reg=reg,
        # Number of iteration for the alternate minimization and cvg threshold
        n_iter=n_iter,  # original value: 100
        eps=eps,  # original value: 1e-4
        # solver for the z-step
        solver_z="lgcd",
        solver_z_kwargs={'tol': 1e-3,  # stopping criteria
                         'max_iter': 100000},
        # solver for the d-step
        solver_d='alternate_adaptive',
        solver_d_kwargs={'max_iter': 300},  # original value: 300
        # sort atoms by explained variances
        sort_atoms=True,
        # Technical parameters
        verbose=1,
        random_state=0,
        n_jobs=n_jobs)

    # Fit the model
    cdl.fit(X_split)
    z_hat = cdl.transform(X[None, :])  # compute atoms activation intensities

    # Duration of the experiment, in seconds
    T = z_hat.shape[2] / sfreq

    # Determine events timestamps and activation vectors
    event_id = data_utils['event_id']
    events_timestamps = utils.get_events_timestamps(events=events,
                                                    sfreq=sfreq,
                                                    event_id=event_id)
    acti_shift = utils.get_activation(model=cdl,
                                      z_hat=z_hat.copy(),
                                      shift=True)
    acti_not_shift = z_hat[0].copy()

    # Construct parameters dictionaries and save the model
    dict_cdl_params = utils.get_dict_cdl_params(cdl)

    dict_other_params = {'data_source': 'sample',
                         'file_name': file_name,
                         'subject': 'sample',
                         'stim_channel': stim_channel,
                         'sfreq': sfreq,
                         'n_splits': n_splits,
                         'event_id': event_id}

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

    # save results in JSON file
    json_file_path = SAVE_RESULTS_PATH / ('cdl_' + data_source + '.json')
    with open(json_file_path, 'w') as fp:
        json.dump(dict_global, fp, sort_keys=True, indent=4,
                  cls=utils.NumpyEncoder)

    return dict_global


def run_cdl_sample(sfreq=150., n_atoms=40, n_times_atom=None, reg=0.1,
                   n_iter=100, eps=1e-4, n_jobs=5, n_splits=10):
    """Run Convolutional Dictionary Learning on mne.sample

    """
    return _run_cdl_data(sfreq=sfreq, n_atoms=n_atoms,
                         n_times_atom=n_times_atom, reg=reg,
                         n_iter=n_iter, eps=eps, n_jobs=n_jobs,
                         n_splits=n_splits,
                         data_source='sample')


def run_cdl_somato(sfreq=150., n_atoms=25, n_times_atom=None, reg=0.2,
                   n_iter=100, eps=1e-4, n_jobs=5, n_splits=10):
    """Run Convolutional Dictionary Learning on mne.somato

    """
    return _run_cdl_data(sfreq=sfreq, n_atoms=n_atoms,
                         n_times_atom=n_times_atom, reg=reg,
                         n_iter=n_iter, eps=eps, n_jobs=n_jobs,
                         n_splits=n_splits,
                         data_source='somato')
