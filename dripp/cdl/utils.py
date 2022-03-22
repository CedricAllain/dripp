# %%
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import mne
from mne_bids import BIDSPath, read_raw_bids

# for Cam-CAN dataset
DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

# %%
###############################################################################
# General
###############################################################################


def get_data_utils(data_source='sample', subject='sample',
                   verbose=True):
    """Returns dataset's informations such as paths, STIM channel,
    events description, etc.

    Parameters
    ----------
    data_source : 'sample' | 'camcan'
        Name of the dataset. Defaults to 'sample'.

    subject : str
        Subject label. Defaults to 'sample'.

    verbose : bool
        If True, print events description. Defaults to True.

    Returns
    -------
    dict
    """

    if data_source == 'sample':
        data_path = mne.datasets.sample.data_path()
        data_folder = os.path.join(data_path, 'MEG', 'sample')
        file_name = os.path.join(data_folder, 'sample_audvis_raw.fif')
        # Path to BEM solution
        subjects_dir = os.path.join(data_path, 'subjects')
        fname_bem = os.path.join(subjects_dir, 'sample', 'bem',
                                 'sample-5120-bem-sol.fif')
        # Path to transformation
        fname_trans = os.path.join(data_folder,
                                   'sample_audvis_raw-trans.fif')
        # Path to noise covariance matrix
        fname_cov = os.path.join(data_folder,
                                 'sample_audvis-cov.fif')
        # Other
        stim_channel = 'STI 014'  # STIM channel
        bads = []  # bad chanels
        add_bads = False
        event_id = [1, 2, 3, 4]  # event id to keep for evoking
        event_des = {'auditory/left': 1, 'auditory/right': 2,
                     'visual/left': 3, 'visual/right': 4,
                     'smiley': 5, 'buttonpress': 32}

    elif data_source == 'somato':
        data_path = mne.datasets.somato.data_path()
        subjects_dir = None
        file_name = os.path.join(data_path, 'sub-01', 'meg',
                                 'sub-01_task-somato_meg.fif')
        event_id = [1]
        event_des = {'somato': 1}
        stim_channel = 'STI 014'  # STIM channel
        bads = []  # bad chanels
        add_bads = False

        fname_cov = None
        fname_trans = None
        fname_bem = None

    elif data_source == 'camcan':
        event_id = [1, 2, 3, 4, 5, 6]
        event_des = {'audiovis/1200Hz': 1,  # bimodal
                     'audiovis/300Hz': 2,   # bimodal
                     'audiovis/600Hz': 3,   # bimodal
                     'button': 4,
                     'catch/0': 5,          # unimodal
                     'catch/1': 6}          # unimodal
    else:
        raise ValueError("data source %s is unknown" % data_source)

    if verbose:
        print("Events description:", event_des)
        print("Only consider events", event_id)

    data_utils = {'event_id': event_id, 'event_des': event_des}
    if data_source in ['sample', 'somato']:
        data_utils = dict(data_utils, **{'file_name': file_name,
                                         'fname_bem': fname_bem,
                                         'fname_trans': fname_trans,
                                         'fname_cov': fname_cov,
                                         'stim_channel': stim_channel,
                                         'subject': subject,
                                         'bads': bads, 'add_bads': add_bads})

    return data_utils


def get_subject_info(subject_id, verbose=True):
    """
    """

    # get age and sex of the subject
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
    age, sex = participants[participants['participant_id']
                            == 'sub-' + str(subject_id)][['age', 'sex']].iloc[0]
    if verbose:
        print(f'Subject ID: {subject_id}, {str(age)} year old {sex}')

    return age, sex


def get_info_camcan(subject_id):
    # get age and sex of the subject
    get_subject_info(subject_id)
    raw, events, _, _ = raw_preprocessing(data_source='camcan')
    print('Experiment duration, %.3f seconds' %
          raw.get_data(picks=['meg']).shape[1] / 150.)
    print('Counter events: ', Counter(events[:, -1]))


def raw_preprocessing(data_source, subject_id=None, sfreq=150.):
    """For a given dataset name, apply a specific pre-processing and search for
    events.

    Parameters
    ----------
    data_source : 'sample' | 'somamto' | 'camcan'
        Name of the dataset. Defaults to 'sample'.

    subject_id : str | None
        For Cam-CAN dataset, the subject id to run the CSC on.
        Defaults to None.
        Ex.: 'CC620264', a 76.33 year old woman.

    sfreq : double
        Sampling frequency. The signal will be resampled to match this.
        Defaults to 150.

    Returns
    -------
    raw : instance of mne.Raw

    events : 2d array

    event_id : list

    event_des : dict
    """

    # get dataset utils
    data_utils = get_data_utils(data_source=data_source, verbose=True)

    # Load data and preprocessing
    print("Loading the data...", end=' ', flush=True)
    if data_source in ['sample', 'somato']:
        file_name = data_utils['file_name']
        raw = mne.io.read_raw_fif(file_name, preload=True, verbose=False)
        raw.pick_types(meg='grad', eeg=False, eog=False, stim=True)
    elif data_source == 'camcan':
        assert subject_id is not None

        bp = BIDSPath(
            root=BIDS_ROOT,
            subject=subject_id,
            task="smt",
            datatype="meg",
            extension=".fif",
            session="smt",
        )
        raw = read_raw_bids(bp)
    print('done')

    print("Preprocessing the data...", end=' ', flush=True)
    if data_source == 'sample':
        raw.notch_filter(np.arange(60, 181, 60))
        raw.filter(l_freq=2, h_freq=None)
    elif data_source == 'somato':
        raw.notch_filter(np.arange(50, 101, 50))
        raw.filter(l_freq=2, h_freq=None)
    elif data_source == 'camcan':
        raw.load_data()
        raw.filter(l_freq=None, h_freq=125)
        raw.notch_filter([50, 100])
        raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                               cross_talk=CT_SPARSE_FILE,
                                               st_duration=10.0)

    if data_source in ['sample', 'somato']:
        stim_channel = data_utils['stim_channel']
        events = mne.find_events(raw, stim_channel=stim_channel)
        event_id = data_utils['event_id']
        event_des = data_utils['event_des']
    elif data_source == 'camcan':
        raw.pick(['grad', 'stim'])
        events, event_des = mne.events_from_annotations(raw)
        # event_id = {'audiovis/1200Hz': 1,
        #             'audiovis/300Hz': 2,
        #             'audiovis/600Hz': 3,
        #             'button': 4,
        #             'catch/0': 5,
        #             'catch/1': 6}
        event_id = list(event_des.values())
        raw.filter(l_freq=2, h_freq=45)

    raw, events = raw.resample(
        sfreq, npad='auto', verbose=False, events=events)
    # Set the first sample to 0 in event stim
    events[:, 0] -= raw.first_samp
    print('done')

    return raw, events, event_id, event_des

# %%

###############################################################################
# On driver's events
###############################################################################


def get_event_id_from_type(event_des=None, event_type='all', data_source=None,
                           verbose=True):
    """Get the event id, or ids, corresponding to a type,
    given an events description.

    Parameters
    ----------
    event_des : dict
        Events descriptions. Keys are event type, e.g., 'visual/left', values
        are corresponding event ids. Note that in type, specifications must be
        separated by a '/'. Defaults to None.

    event_type : str | list of str | 'all'
        Type(s) of event(s) we want to select. If 'all', all events id are
        returned. Defaults to'all'.
        Exemples:
            event_type = "visual/left" -> return the corresponding id in a list
            of length 1.
            event_type = ”visual" -> return a list of length 1, containing a
            sub-list of all id that have the specification.
            event_type = ["visual", "auditory"] -> return a list of length 2,
            containing 2 sub-lists of id, one sub-list per specification.

    data_source : 'sample' | 'camcan'
        Name of the dataset, to determine event_des if the latter is None.
        Defaults to 'sample'.

    verbose : bool
        If True, print some info. Defaults to True.

    Returns
    -------
    list of int | list of list of int
    """
    assert (event_des is not None) or (data_source is not None), \
        "No event descriptions nor data source is given."

    if event_des is None:
        # determine event descriptions based on data source
        data_utils = get_data_utils(
            data_source, subject='CC110033', kind='passive', verbose=False)
        event_des = data_utils['event_des']

    if event_type == 'all':
        return list(event_des.values())

    if isinstance(event_type, str):
        event_type = [event_type]

    event_id = []
    for t in event_type:
        id_temp = []
        for k, v in event_des.items():
            if k == t:
                id_temp.append(v)
                break

            k_split = k.split('/')
            if t in set(k_split):
                id_temp.append(v)

        if len(id_temp) == 1:
            event_id.append(id_temp[0])
        elif len(id_temp) >= 2:
            event_id.append(id_temp)
        elif verbose:
            print('No event of type %s in the given event description.' % t)

    return event_id


def get_event_type_from_id(event_des_reverse=None, event_id=None,
                           data_source='sample'):
    """Get the event type, or types, corresponding to an id(s),
    given an inverse events description.

    Parameters
    ----------
    event_des_reverse : dict
        Inverse events descriptions. Keys are event id, values are event type,
        e.g., 'visual/left'. Note that in type, specifications must be
        separated by a '/'. Defaults to None.

    event_id : int | list of int | list of list of int
        Id of events for which to find the corresponding type.
        Exemples:
            event_id = 1 -> return the type corresponding in a list of length 1
            event_id = [1, 2, 3, 4] -> return a list of the 4 types
            event_id = [[1, 2], [3, 4]] -> return a list of length 2, for each
            sublist of event id, return the common specification

    data_source : 'sample' | 'camcan'
        Name of the dataset, to determine event_des if the latter is None.
        Defaults to 'sample'.

    Returns
    -------
    list of str
    """
    assert (event_des_reverse is not None) or (data_source is not None), \
        "No event descriptions nor data source is given."

    if event_des_reverse is None:
        # determine event descriptions based on data source
        data_utils = get_data_utils(
            data_source, subject='CC110033', kind='passive', verbose=False)
        event_des_reverse = {v: k for k, v in data_utils['event_des'].items()}

    if isinstance(event_id, int):
        if event_id in event_des_reverse.keys():
            return [event_des_reverse[event_id]]
        else:
            return []

    if isinstance(event_id, (list, np.array)):
        labels = []
        for this_id in event_id:
            if isinstance(this_id, int):
                labels.append(event_des_reverse[this_id])
                continue

            # this_id is a list of event ids
            temp = []
            for i in this_id:
                temp.append(event_des_reverse[i])

            if len(temp) == 1:
                labels.append(temp[0])

            # find common specification
            common = set(temp[0].split('/'))
            for label in temp[1:]:
                common = common & set(label.split('/'))

            labels.append(list(common)[0])

    return labels


def get_events_timestamps(events=None, sfreq=None, info=None,
                          event_id='all'):
    """Return the dictionary of the timestamps corresponding to a set of event
    ids.

    Parameters
    ----------
    events : 2d-array of shape (n_events, 3)
        The events array, as used in MNE. If None, will search for an "events"
        key in the info dictionary. Defaults to None.

    sfreq : float
        The sampling frequency. If None, will search for an "sfreq" key in the
        info dictionary. Defaults to None.

    info : dict
        Dictionary containing information about the experiment, similar to
        mne.Info. Defaults to None.

    event_id : 'all' | list of int
        List of event id for which to compute their timestamps. If 'all', all
        event ids are considered. Defaults to'all'.

    Returns
    -------
    events_timestamps : dict
        Keys are int, the event id, and values are numpy.array of float, the
        event's timestamps (in seconds).
    """

    if events is None:
        events = info['events']

    if sfreq is None:
        sfreq = info['sfreq']

    if event_id == 'all':
        event_id = list(set(events[:, -1]))

    events_timestamps = {}  # save events' timestamps in a dictionary

    for i in event_id:
        mask = events[:, -1] == i
        events_timestamps[i] = events[:, 0][mask] / sfreq

    return events_timestamps


###############################################################################
# On stochastic process' activations
###############################################################################


def get_activation(model, z_hat=None, idx_atoms='all', shift=True):
    """Get activation sparse vector from CDL results.

    Parameters
    ----------
    model : alphacsc.convolutional_dictionary_learning.GreedyCDL | dict
        Fitted model. If dict, must have v_hat_ and z_hat in its keys.

    z_hat : numpy.array of shape (n_trials, n_atoms, n_timestamps)
        If z_hat is None, model must be a dict containing a 'z_hat' key.
        Defaults to None

    idx_atoms : int | list of int | 'all'
        The idices of the atoms to consider. If 'all', then all the extracted
        atoms are taken. Defaults to 'all'.

    shift : bool
        If True, apply, for each atom's ativations, a shift of size equal.
        Defaults to True.

    Returns
    -------
    acti : numpy array
        shape (n_atoms, T * sfreq)
        the sparse vectors of atoms' activations values
    """
    # retrieve fitting results
    if isinstance(model, dict):
        v_hat_ = np.array(model['v_hat_'])
        z_hat = np.array(model['z_hat'])
    else:
        v_hat_ = model.v_hat_

    assert z_hat is not None, \
        "if z_hat is None, model must be a dict containing a 'z_hat' key"

    if isinstance(idx_atoms, str) and idx_atoms == 'all':
        idx_atoms = list(range(z_hat.shape[1]))
    elif isinstance(idx_atoms, int):
        idx_atoms = [idx_atoms]

    assert isinstance(idx_atoms, list), \
        "idx_atoms must be 'all' or of type int | list of int "

    # select desired atoms
    acti = z_hat[0, idx_atoms]

    if shift:
        # roll to put activation to the peak amplitude time in the atom.
        for kk in idx_atoms:
            shift = np.argmax(np.abs(v_hat_[kk]))
            acti[kk] = np.roll(acti[kk], shift)
            acti[kk][:shift] = 0  # pad with 0

    return acti


def block_process_1d(a, blocksize):
    """For a given array a, returns an array of same size b, but with only the
    constructed by keeping maximum values of a within blocks of given size.

    Parameters
    ----------
    a : numpy.array
        Array to process.

    blocksize : int
        Size of the block to process a with.

    Returns
    -------
    b : numpy.array
        Processed array, of same shape of input array a.

    Examples
    --------
    >>> a = numpy.array([0, 1, 0, 0, 1, 3, 0])
    >>> blocksize = 2
    >>> block_process_1d(a, blocksize)
    numpy.array([0, 1, 0, 0, 0, 3, 0])
    """
    if len(a) < blocksize:
        return a

    b = np.zeros(a.shape)
    a_len = a.shape[0]
    for i in range(a_len):
        block = a[int(max(i-blocksize+1, 0)): int(min(i+blocksize, a_len))]
        if np.max(block) == a[i]:
            b[i] = a[i]

    return b


def filter_activation(acti, atom_to_filter='all', sfreq=150.,
                      time_interval=0.01):
    """For an array of atoms activations values, only keeps maximum values
    within a given time intervalle.

    In other words, we apply a filter in order to have a minimum time
    intervalle between two consecutives activations, and only keeping the
    maximum values

    Parameters
    ----------
    acti : numpy.array

    atom_to_filter : 'all' | int | array-like of int
        Ids of atoms to apply the filter on. If 'all', then applied on every
        atom in input `acti`. Defaults to 'all'.

    sfreq = float
        Sampling frequency, allow to transform `time_interval` into a number of
        timestamps. Defaults to 150.

    time_interval : float
        In second, the time interval within which we would like to keep the
        maximum activation values. Defaults to 0.01

    Returns
    -------
    acti : numpy.array
        Same as input, but with only maximum values within the given time
        intervalle.
    """

    blocksize = round(time_interval * sfreq)
    print("Filter activation on {} atoms using a sliding block of {} "
          "timestamps.".format(
              atom_to_filter, blocksize))

    if isinstance(atom_to_filter, str) and atom_to_filter == 'all':
        acti = np.apply_along_axis(block_process_1d, 1, acti, blocksize)
    elif isinstance(atom_to_filter, (list, np.ndarray)):
        for aa in atom_to_filter:
            acti[aa] = block_process_1d(acti[aa], blocksize)
    elif isinstance(atom_to_filter, int):
        acti[atom_to_filter] = block_process_1d(
            acti[atom_to_filter], blocksize)

    return acti


def get_atoms_timestamps(acti, sfreq=None, info=None, threshold=0,
                         percent=False, per_atom=True):
    """Get atoms' activation timestamps, using a threshold on the activation
    values to filter out unsignificant values.

    Parameters
    ----------
    acti : numpy.array of shape (n_atoms, n_timestamps)
        Sparse vector of activations values for each of the extracted atoms.

    sfreq : float
        Sampling frequency used in CDL. If None, will search for an "sfreq" key
        in the info dictionary. Defaults to None.

    info : dict
        Similar to mne.Info instance. Defaults to None.

    threshold : int | float
        Threshold value to filter out unsignificant ativation values. Defaults
        to 0.

    percent : bool
        If True, threshold is treated as a percentage: e.g., threshold = 5
        indicates that 5% of the activations will be removed, either per atom,
        or globally. Defaults to False.

    per_atom : bool
        If True, the threshold as a percentage will be applied per atom, e.g., 
        threshold = 5 will remove 5% of the activation of each atom. If false,
        the thresholding will be applied to all the activations. Defaults to
        True.

    Returns
    -------
    atoms_timestamps : numpy array
        array of timestamps
    """

    assert (sfreq is not None) or ('sfreq' in info.keys()), \
        "Please give an info dict that has a 'sfreq' key."

    if sfreq is None:
        sfreq = info['sfreq']

    n_atoms = acti.shape[0]
    if percent and per_atom:
        acti_nan = acti.copy()
        acti_nan[acti_nan == 0] = np.nan
        mask = acti_nan >= np.nanpercentile(
            acti_nan, threshold, axis=1, keepdims=True)
        atoms_timestamps = [acti_nan[i][mask[i]] / sfreq
                            for i in range(n_atoms)]
        return atoms_timestamps

    if percent and not per_atom:
        # compute the q-th percentile over all positive values
        threshold = np.percentile(acti[acti > 0], threshold)

    atoms_timestamps = [np.where(acti[i] > threshold)[0] / sfreq
                        for i in range(n_atoms)]

    return atoms_timestamps


###############################################################################
# Post-processing
###############################################################################

def cdl_postprocess(cdl, z_hat, events, event_id, sfreq=150., shift=True,
                    threshold=0):
    T = z_hat.shape[2] / sfreq

    # Determine events timestamps and activation vectors
    events_tt = get_events_timestamps(
        events=events, sfreq=sfreq, event_id=event_id)
    if shift:
        acti = get_activation(model=cdl, z_hat=z_hat.copy(), shift=True)
    else:
        acti = z_hat[0].copy()

    acti = filter_activation(acti, sfreq)

    acti_tt = get_atoms_timestamps(acti=acti, sfreq=sfreq, threshold=threshold)

    return acti_tt, events_tt, T

###############################################################################
# To save variables, parameters and results into a JSON file
###############################################################################


def get_dict_cdl_params(cdl):
    """From a alphacsc.GreedyCDL instance, returns a dictionary with its main
    parameters so it can be savec in a JSON file

    Parameters
    ----------
    cdl : alphacsc.GreedyCDL instance

    Returns
    -------
    dict_cdl_params : dict
    """
    dict_cdl_params = {'n_atoms': cdl.n_atoms,
                       'n_times_atom': cdl.n_times_atom,
                       'rank1': cdl.rank1,
                       'uv_constraint': cdl.uv_constraint,
                       'window': cdl.window,
                       'unbiased_z_hat': cdl.unbiased_z_hat,
                       'D_init': cdl.D_init,
                       'lmbd_max': cdl.lmbd_max,
                       'reg': cdl.reg,
                       'n_iter': cdl.n_iter,
                       'eps': cdl.eps,
                       'solver_z': cdl.solver_z,
                       'solver_z_kwargs': cdl.solver_z_kwargs,
                       'solver_d': cdl.solver_d,
                       'solver_d_kwargs': cdl.solver_d_kwargs,
                       'sort_atoms': cdl.sort_atoms,
                       'verbose': cdl.verbose,
                       'random_state': cdl.random_state,
                       'n_jobs': cdl.n_jobs}

    return dict_cdl_params


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
