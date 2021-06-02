import os
import json
from datetime import datetime

import numpy as np
import mne


from ..base import NumpyEncoder


###############################################################################
# General
###############################################################################

def get_data_utils(data_source='sample', subject='sample', kind='passive',
                   verbose=True):
    """ Returns dataset's informations such as paths, STIM channel,
    events description, etc.

    Parameters
    ----------
    data_source : 'sample' | 'camcan'
        Name of the dataset, default is 'sample'

    subject : str
        subject label for camcan dataset, e.g., subject = 'CC110033'
        default is 'sample'

    kind : 'passive' | 'rest' | 'task'
        only for camcan dataset, kind of experiment done on the subject
        default is 'passive'

    verbose : bool
        if True, print events description

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
        if not subject != 'sample':
            raise ValueError("No subject label is given while using "
                             "camcan dataset. Please specify a label, "
                             "such as 'CC110033'.")

        data_folder = r'/storage/store/data/camcan/camcan47/cc700/meg/pipeline\
                        /release004/data/aamod_meg_get_fif_00001/'
        file_name = os.path.join(data_folder, subject, kind, kind + '_raw.fif')
        # Path to BEM solution
        subjects_dir = r'/storage/store/data/camcan-mne/freesurfer/'
        fname_bem = os.path.join(subjects_dir, subject, 'bem',
                                 subject + '-meg-bem.fif')
        # Path to transformation
        trans_folder = r'/storage/store/data/camcan-mne/trans/'
        fname_trans = os.path.join(trans_folder,
                                   'sub-' + subject + '-trans.fif')
        # Path to noise covariance matrix
        fname_cov = None
        # Other
        stim_channel = 'STI101'  # STIM channel
        bads = ['MEG2113']  # bad chanels
        add_bads = True
        event_id = [6, 7, 8, 9]  # event id to keep for evoking
        event_des = {"auditory/300Hz": 6, "auditory/600Hz": 7,
                     "auditory/1200Hz": 8, "Visual Checkerboard": 9}
    else:
        raise ValueError("data source %s is unknown" % data_source)

    if verbose:
        print("Events description:", event_des)
        print("Only consider events", event_id)

    data_utils = {'file_name': file_name,
                  'fname_bem': fname_bem,
                  'fname_trans': fname_trans,
                  'fname_cov': fname_cov,
                  'stim_channel': stim_channel,
                  'subject': subject,
                  'bads': bads, 'add_bads': add_bads,
                  'event_id': event_id, 'event_des': event_des}

    return data_utils


###############################################################################
# On events
###############################################################################

def get_event_id_from_type(event_des=None, event_type='all', data_source=None,
                           verbose=True):
    """ Get the event id, or ids, corresponding to a type,
    given an events description.

    Parameters
    ----------
    event_des : dict
        events descriptions
        keys are event type, e.g., 'visual/left',
        values are event id
        note that in type, specifications must be separated by a '/'
        default is None

    event_type : str | list of str | 'all'
        type(s) of event(s) we want to select
        if 'all', all events id are returned
        default is 'all'
        exemples:
            event_type = "visual/left" -> return the corresponding id in a list
            of length 1
            event_type = ”visual" -> return a list of length 1, containing a
            sub-list of all id that have the specification
            event_type = ["visual", "auditory"] -> return a list of length 2,
            containing 2 sub-lists of id, one sub-list per specification

    data_source : 'sample' | 'camcan'
        Name of the dataset, to determine event_des if the latter is None
        default is 'sample'

    verbose : bool
        if True, print some info

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
    """ Get the event type, or types, corresponding to an id(s),
    given an inverse events description.

    Parameters
    ----------
    event_des_reverse : dict
        inverse events descriptions
        keys are event id,
        values are event type, e.g., 'visual/left',
        note that in type, specifications must be separated by a '/'
        default is None

    event_id : int | list of int | list of list of int
        id of events for which to find the corresponding type
        exemples:
            event_id = 1 -> return the type corresponding in a list of length 1
            event_id = [1, 2, 3, 4] -> return a list of the 4 types
            event_id = [[1, 2], [3, 4]] -> return a list of length 2, for each
            sublist of event id, return the common specification

    data_source : 'sample' | 'camcan'
        Name of the dataset, to determine event_des if the latter is None
        default is 'sample'

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
    """

    Parameters
    ----------

    event_id : 'all' | list of int
        list of event id for which to compute their timestamps
        if 'all', all event ids are considered
        default is 'all'

    Returns
    -------
    events_timestamps : dict
        keys are int, the event id,
        values are numpy.array of float, the event's timestamps

    """

    if events is None:
        events = info['events']

    if sfreq is None:
        sfreq = info['sfreq']

    if event_id == 'all':
        event_id = list(set(events[:, -1]))

    # n_events = len(event_id)
    # events_timestamps = np.empty(shape=(n_events), dtype='object')
    events_timestamps = {}  # save events' timestamps in a dictionary

    for i in event_id:
        mask = events[:, -1] == i
        events_timestamps[i] = events[:, 0][mask] / sfreq

    return events_timestamps


###############################################################################
# On activations
###############################################################################


def get_activation(model, z_hat=None, idx_atoms='all', shift=True):
    """ Get activation sparse vector from CDL results.

    Parameters
    ----------
    model : alphacsc.convolutional_dictionary_learning.GreedyCDL | dict
        fitted model
        if dict, must have v_hat_ and z_hat in its keys

    z_hat : numpy.array of shape (n_trials, n_atoms, n_timestamps)

    idx_atoms : int | list of int | 'all'

    shift : bool
        if True, apply, for each atom's ativations, a shift of size equal

    Returns
    -------

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

    # select atom
    acti = z_hat[0, idx_atoms]

    if shift:
        # roll to put activation in the peak of the atoms
        for kk in idx_atoms:
            shift = np.argmax(np.abs(v_hat_[kk]))
            acti[kk] = np.roll(acti[kk], shift)
            acti[kk][:shift] = 0  # pad with 0

    return acti


def block_process_1d(a, blocksize):
    """

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
    """

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


def get_atoms_timestamps(acti, sfreq=None, info=None, threshold=0):
    """ Get atoms' activation timestamps

    Parameters
    ----------

    acti : numpy.array of shape (n_atoms, n_timestamps)
        sparse vector where


    Returns
    -------

    """

    assert (sfreq is not None) or ('sfreq' in info.keys()), \
        "Please give an info dict that has a 'sfreq' key."

    if sfreq is None:
        sfreq = info['sfreq']

    n_atoms = acti.shape[0]
    atoms_timestamps = np.array([np.where(acti[i] > threshold)[0]
                                 for i in range(n_atoms)], dtype="object")
    atoms_timestamps /= sfreq

    return atoms_timestamps


def get_time_array(events_timestamps, first_event=0, n_events_to_plot=6):
    """

    """
    n_events = events_timestamps.shape[0]

    last_event = first_event + n_events_to_plot

    min_timestamp = min([events_timestamps[_][first_event]
                         for _ in range(n_events)])
    max_timestamp = max([events_timestamps[_][last_event]
                         for _ in range(n_events)])

    low_t = int(np.floor(min_timestamp))
    up_t = int(np.ceil(max_timestamp))

    time_array = np.linspace(*(low_t, up_t), (up_t-low_t)*10)

    return time_array


def get_intensity(kernels, baselines, events_timestamps, sfreq=150.,
                  time_array=np.linspace(0, 100, int(1e4)), method='separate'):
    """

    kernels: list of kernels, one per event id

    method: str, 'separate'|'add' (default 'separate')
        if 'separate', keep the intensity function separate per event id
        if 'add', add the intensities functions

    """
    if not isinstance(kernels, (list, np.ndarray)):
        kernel = np.array([kernels])

    if not isinstance(baselines, (list, np.ndarray)):
        baselines = np.array([baselines])

    n_events = events_timestamps.shape[0]

    intensities = []  # np.empty(shape=(n_events,), dtype='object')
    for ii in range(n_events):
        kernel = kernels[ii]
        baseline = baselines[ii]
        event_times = events_timestamps[ii] / sfreq
        n_timestamps = event_times.shape[0]

        intensity = np.array([kernel.get_values(time_array - event_times[_])
                              for _ in range(n_timestamps)])
        # intensities[ii] = baseline + intensity.sum(axis=0)
        intensities.append(baseline + intensity.sum(axis=0))

    intensities = np.array(intensities)

    if method == 'add':
        return intensities.sum(axis=0)
    elif method == 'separate':
        return intensities


###############################################################################
# To save variables, parameters and results into a JSON file
###############################################################################


def get_dict_cdl_params(cdl):
    """

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


def save_dict_global(dict_global, json_file_path):
    """

    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub_dir_path = os.path.join(json_file_path, now)
    if not os.path.isdir(sub_dir_path):
        os.mkdir(sub_dir_path)

    data_source = dict_global['dict_other_params']['data_source']
    json_file_name = 'dict_global_' + data_source + '_' + now + '.json'

    json_file_path = os.path.join(sub_dir_path, json_file_name)

    with open(json_file_path, 'w') as fp:
        json.dump(dict_global, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    print("To read JSON file: \nwith open(json_file_path, 'r') as fp: \
        \n\tdict_global = json.load(fp)")

    return json_file_path
