# %%
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import mne

from dripp.cdl.run_cdl import run_default_cdl

# from mne_bids import BIDSPath, read_raw_bids

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


def get_data_utils(data_source="sample", subject="sample", verbose=True):
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

    if data_source == "sample":
        data_path = mne.datasets.sample.data_path()
        data_folder = os.path.join(data_path, "MEG", "sample")
        file_name = os.path.join(data_folder, "sample_audvis_raw.fif")
        # Path to BEM solution
        subjects_dir = os.path.join(data_path, "subjects")
        fname_bem = os.path.join(
            subjects_dir, "sample", "bem", "sample-5120-bem-sol.fif"
        )
        # Path to transformation
        fname_trans = os.path.join(data_folder, "sample_audvis_raw-trans.fif")
        # Path to noise covariance matrix
        fname_cov = os.path.join(data_folder, "sample_audvis-cov.fif")
        # Other
        stim_channel = "STI 014"  # STIM channel
        bads = []  # bad chanels
        add_bads = False
        event_id = [1, 2, 3, 4]  # event id to keep for evoking
        event_des = {
            "auditory/left": 1,
            "auditory/right": 2,
            "visual/left": 3,
            "visual/right": 4,
            "smiley": 5,
            "buttonpress": 32,
        }

    elif data_source == "somato":
        data_path = mne.datasets.somato.data_path()
        subjects_dir = None
        file_name = os.path.join(
            data_path, "sub-01", "meg", "sub-01_task-somato_meg.fif"
        )
        event_id = [1]
        event_des = {"somato": 1}
        stim_channel = "STI 014"  # STIM channel
        bads = []  # bad chanels
        add_bads = False

        fname_cov = None
        fname_trans = None
        fname_bem = None

    elif data_source == "camcan":
        event_id = [1, 2, 3, 4, 5, 6]
        event_des = {
            "audiovis/1200Hz": 1,  # bimodal
            "audiovis/300Hz": 2,  # bimodal
            "audiovis/600Hz": 3,  # bimodal
            "button": 4,
            "catch/0": 5,  # unimodal
            "catch/1": 6,
        }  # unimodal
    else:
        raise ValueError("data source %s is unknown" % data_source)

    if verbose:
        print("Events description:", event_des)
        print("Only consider events", event_id)

    data_utils = {"event_id": event_id, "event_des": event_des}
    if data_source in ["sample", "somato"]:
        data_utils = dict(
            data_utils,
            **{
                "file_name": file_name,
                "fname_bem": fname_bem,
                "fname_trans": fname_trans,
                "fname_cov": fname_cov,
                "stim_channel": stim_channel,
                "subject": subject,
                "bads": bads,
                "add_bads": add_bads,
            },
        )

    return data_utils


# %%

###############################################################################
# On driver's events
###############################################################################


def get_event_id_from_type(
    event_des=None, event_type="all", data_source=None, verbose=True
):
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
    assert (event_des is not None) or (
        data_source is not None
    ), "No event descriptions nor data source is given."

    if event_des is None:
        # determine event descriptions based on data source
        data_utils = get_data_utils(
            data_source, subject="CC110033", kind="passive", verbose=False
        )
        event_des = data_utils["event_des"]

    if event_type == "all":
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

            k_split = k.split("/")
            if t in set(k_split):
                id_temp.append(v)

        if len(id_temp) == 1:
            event_id.append(id_temp[0])
        elif len(id_temp) >= 2:
            event_id.append(id_temp)
        elif verbose:
            print("No event of type %s in the given event description." % t)

    return event_id


def get_event_type_from_id(event_des_reverse=None, event_id=None, data_source="sample"):
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
    assert (event_des_reverse is not None) or (
        data_source is not None
    ), "No event descriptions nor data source is given."

    if event_des_reverse is None:
        # determine event descriptions based on data source
        data_utils = get_data_utils(
            data_source, subject="CC110033", kind="passive", verbose=False
        )
        event_des_reverse = {v: k for k, v in data_utils["event_des"].items()}

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
            common = set(temp[0].split("/"))
            for label in temp[1:]:
                common = common & set(label.split("/"))

            labels.append(list(common)[0])

    return labels


def get_events_timestamps(events, event_id="all", sfreq=1.0):
    """Return the dictionary of the timestamps corresponding to a set of event
    ids.

    Parameters
    ----------
    events : 2d-array of shape (n_events, 3)
        The events array, as used in MNE.

    event_id : 'all' | list of int | list of tuples
        List of event id for which to compute their timestamps. If 'all', all
        event ids are considered. Defaults to 'all'.

    sfreq : float
        The sampling frequency.
        Defaults to 1, i.e., return vector positions.

    Returns
    -------
    events_timestamps : dict
        Keys are int, the event id, and values are numpy.array of float, the
        event's timestamps (in seconds).
    """

    if event_id == "all":
        event_id = list(set(events[:, -1]))

    if isinstance(event_id, int):
        event_id = np.atleast_1d(event_id)

    if not isinstance(event_id, dict):
        event_id = {i: v for i, v in enumerate(event_id)}

    events_tt = {}  # save events' timestamps in a dictionary

    def proc(evt_id):
        if evt_id in events_tt.keys():
            return events_tt[evt_id]
        else:
            mask = events[:, -1] == evt_id
            return events[:, 0][mask] / sfreq

    for label, this_id in event_id.items():
        if isinstance(this_id, int):
            events_tt[label] = proc(this_id)

        elif isinstance(this_id, (tuple, list, np.ndarray)):
            tt = np.concatenate([proc(evt_id) for evt_id in this_id])
            tt.sort()
            events_tt[label] = tt

    return events_tt


###############################################################################
# On stochastic process' activations
###############################################################################


def get_activation(z_hat, idx_atoms="all", shift=True, v_hat_=None):
    """Get activation sparse vector from CDL results.

    Parameters
    ----------
    z_hat : numpy.array of shape (n_trials, n_atoms, n_timestamps)

    idx_atoms : int | list of int | 'all'
        The idices of the atoms to consider. If 'all', then all the extracted
        atoms are taken. Defaults to 'all'.

    shift : bool
        If True, apply, for each atom's ativations, a shift of size equal.
        Defaults to True.

    v_hat_ : numpy.ndarray of shape (n_atoms, n_times_atom)

    Returns
    -------
    acti : numpy array
        shape (n_atoms, n_times)
        the sparse vectors of atoms' activations values
    """

    if shift:
        assert v_hat_ is not None, "v_hat_ is needed in order to shift activations."

    n_atoms = z_hat.shape[1]
    if idx_atoms == "all":
        idx_atoms = np.array(range(n_atoms))
    elif isinstance(idx_atoms, (int, list, np.ndarray)):
        idx_atoms = np.atleast_1d(idx_atoms)

    assert (
        idx_atoms >= n_atoms
    ).sum() == 0, f"idx_atoms must contain values < {n_atoms}"

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
        block = a[int(max(i - blocksize + 1, 0)) : int(min(i + blocksize, a_len))]
        if np.max(block) == a[i]:
            b[i] = a[i]

    return b


def filter_activation(acti, atom_to_filter="all", sfreq=150.0, time_interval=0.01):
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

    if time_interval is None:
        return acti

    blocksize = round(time_interval * sfreq)
    print(
        "Filter activation on {} atoms using a sliding block of {} "
        "timestamps.".format(atom_to_filter, blocksize)
    )

    if isinstance(atom_to_filter, str) and atom_to_filter == "all":
        acti = np.apply_along_axis(block_process_1d, 1, acti, blocksize)
    elif isinstance(atom_to_filter, (list, np.ndarray)):
        for aa in atom_to_filter:
            acti[aa] = block_process_1d(acti[aa], blocksize)
    elif isinstance(atom_to_filter, int):
        acti[atom_to_filter] = block_process_1d(acti[atom_to_filter], blocksize)

    return acti


def get_activations_timestamps(
    acti, sfreq=None, info=None, threshold=0, percent=False, per_atom=True
):
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

    assert (sfreq is not None) or (
        "sfreq" in info.keys()
    ), "Please give an info dict that has a 'sfreq' key."

    if sfreq is None:
        sfreq = info["sfreq"]

    n_atoms = acti.shape[0]
    if percent and per_atom:
        acti_nan = acti.copy()
        acti_nan[acti_nan == 0] = np.nan
        mask = acti_nan >= np.nanpercentile(acti_nan, threshold, axis=1, keepdims=True)
        activations_tt = [np.where(mask[i])[0] / sfreq for i in range(n_atoms)]
        return activations_tt

    if percent and not per_atom:
        # compute the q-th percentile over all positive values
        threshold = np.percentile(acti[acti > 0], threshold)

    activations_tt = [np.where(acti[i] > threshold)[0] / sfreq for i in range(n_atoms)]

    return activations_tt


###############################################################################
# Global post-processing
###############################################################################


def post_process_cdl(
    events, v_hat_, z_hat, event_id="all", sfreq=1.0, post_process_params=None
):
    """
    XXX
    """

    if post_process_params is None:
        post_process_params = dict(
            time_interval=0.01, threshold=0, percent=True, per_atom=True
        )

    events_tt = get_events_timestamps(events=events, event_id=event_id, sfreq=sfreq)

    acti = filter_activation(
        acti=get_activation(z_hat=z_hat, v_hat_=v_hat_),
        sfreq=sfreq,
        time_interval=post_process_params.pop("time_interval"),
    )
    activations_tt = get_activations_timestamps(
        acti=acti, sfreq=sfreq, **post_process_params
    )

    return events_tt, activations_tt


def get_dict_global(
    dataset="sample",
    subject_id=None,  # for Cam-CAN dataset
    cdl_params={},
    post_process_params=dict(
        time_interval=0.01, threshold=0, percent=True, per_atom=True
    ),
):
    """

    cdl_params : dict
        parameters for CDL model
        By default, is empty, i.e., CDL will run with defaults parameters

    post_process_params : dict

    """

    dict_cdl_res = run_default_cdl(
        dataset, subject_id=subject_id, cdl_params=cdl_params
    )

    sfreq = dict_cdl_res["sfreq"]
    events, event_id = dict_cdl_res["events"], dict_cdl_res["event_id"]
    u_hat_, v_hat_ = dict_cdl_res["u_hat_"], dict_cdl_res["v_hat_"]
    z_hat = dict_cdl_res["z_hat"]

    # Determine events timestamps and activation vectors
    events_tt, activations_tt = post_process_cdl(
        events,
        v_hat_,
        z_hat,
        event_id=event_id,
        sfreq=sfreq,
        post_process_params=post_process_params,
    )

    dict_global = {
        "u_hat_": u_hat_.tolist(),
        "v_hat_": v_hat_.tolist(),
        "z_hat": z_hat.tolist(),
        "T": dict_cdl_res["T"],
        "sfreq": sfreq,
        "events": events,
        "event_id": event_id,
        "events_tt": events_tt,
        "activations_tt": activations_tt,
    }

    return dict_global


###############################################################################
# Others
###############################################################################


def unsplit_z(z):
    """
    Reshape and unsplit the input tensor along the time dimension.

    Parameters
    ----------
    z : ndarray, shape (n_splits, n_atoms, n_times_split)
        The input tensor to be reshaped and unsplit.

    Returns
    -------
    z : ndarray, shape (1, n_atoms, n_times_split * n_splits)
        The unsplit tensor.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.random.rand(2, 3, 4)
    >>> unsplit_z(z).shape
    (1, 3, 8)
    """
    n_splits, n_atoms, n_times_split = z.shape
    z = z.swapaxes(0, 1).reshape(1, n_atoms, n_times_split * n_splits)

    return z


def apply_threshold(z, p_threshold, per_atom=True):
    if len(z.shape) == 3:
        if z.shape[0] > 1:
            z = unsplit_z(z)
        z = z[0]

    n_atoms = z.shape[0]

    if per_atom:
        z_nan = z.copy()
        z_nan[z_nan == 0] = np.nan
        mask = z_nan >= np.nanpercentile(z_nan, p_threshold, axis=1, keepdims=True)

        return [z_nan[i][mask[i]] for i in range(n_atoms)]

    else:
        threshold = np.percentile(z[z > 0], p_threshold)  # global threshold
        print(f"Global thresholding at {p_threshold}%: {threshold}")
        return [this_z[this_z > threshold] for this_z in z]
