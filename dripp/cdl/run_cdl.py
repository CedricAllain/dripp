"""
Run Convolutional Dictionary Learning on mne.sample or mne.somato dataset
"""
import numpy as np

from joblib import Memory

from alphacsc import GreedyCDL, BatchCDL
from alphacsc.datasets.mne_data import load_data as load_data_mne

try:
    from alphacsc.datasets.camcan import load_data as load_data_camcan
except (ValueError, ImportError):
    # temporary, until alphacsc PR #102 is accepted
    from .camcan import load_data as load_data_camcan

from dripp.config import CACHEDIR, BIDS_root, SSS_CAL, CT_SPARSE

memory = Memory(CACHEDIR, verbose=0)


@memory.cache(ignore=["n_jobs"])
def _run_cdl_data(
    dataset="sample",
    subject_id=None,
    load_params=dict(sfreq=150.0),
    use_greedy=True,
    n_atoms=40,
    n_times_atom=None,
    reg=0.1,
    n_iter=100,
    eps=1e-4,
    tol_z=1e-3,
    n_jobs=5,
):
    """Run a Greedy Convolutional Dictionary Learning on mne.[data_source]
    dataset.

    Parameters
    ----------
    dataset : str, 'sample' | 'camcan' | 'somato'
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

    print("Loading and preprocessing the data...", end=" ", flush=True)

    if dataset in ["sample", "somato"]:
        X_split, info = load_data_mne(dataset=dataset, **load_params)

        if dataset == "sample":
            try:
                info["temp"]["event_id"].update({"auditory": (1, 2), "visual": (3, 4)})
            except KeyError:  # temporary, until alphacsc PR #102 is accepted
                info["temp"]["event_id"] = {
                    "auditory/left": 1,
                    "auditory/right": 2,
                    "visual/left": 3,
                    "visual/right": 4,
                    "auditory": (1, 2),  # both auditory event types
                    "visual": (3, 4),  # both visual event types
                    "smiley": 5,
                    "buttonpress": 32,
                }

    elif dataset == "camcan":
        X_split, info = load_data_camcan(
            BIDS_root, SSS_CAL, CT_SPARSE, subject_id, **load_params
        )
        info["temp"]["event_id"].update(
            {
                "audio": (1, 2, 3, 5),  # bimodals events + unimodal auditory
                "vis": (1, 2, 3, 6),  # bimodals events + unimodal visual
            }
        )

    print("done")

    sfreq = load_params["sfreq"]
    if n_times_atom is None:
        # by default, extract atoms of duration 1 second
        n_times_atom = int(round(sfreq * 1.0))

    # Define Greedy Convolutional Dictionary Learning model
    cdl_params = {
        # Shape of the dictionary
        "n_atoms": n_atoms,
        "n_times_atom": n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        "rank1": True,
        "uv_constraint": "separate",
        # apply a temporal window reparametrization
        "window": True,
        # at the end, refit the activations with fixed support
        # and no reg to unbias
        "unbiased_z_hat": True,
        # Initialize the dictionary with random chunk from the data
        "D_init": "chunk",
        # rescale the regularization parameter to be a percentage of lambda_max
        "lmbd_max": "scaled",  # original value: "scaled"
        "reg": reg,
        # Number of iteration for the alternate minimization and cvg threshold
        "n_iter": n_iter,  # original value: 100
        "eps": eps,  # original value: 1e-4
        # solver for the z-step
        "solver_z": "lgcd",
        "solver_z_kwargs": {"tol": tol_z, "max_iter": 100000},  # stopping criteria
        # solver for the d-step
        "solver_d": "alternate_adaptive",
        "solver_d_kwargs": {"max_iter": 300},  # original value: 300
        # sort atoms by explained variances
        "sort_atoms": True,
        # technical parameters
        "verbose": 1,
        "random_state": 0,
        "n_jobs": n_jobs,
    }

    if use_greedy:
        cdl = GreedyCDL(**cdl_params)
    else:
        cdl = BatchCDL(**cdl_params)

    # fit cdl model
    cdl.fit(X_split)
    u_hat_, v_hat_ = cdl.u_hat_, cdl.v_hat_
    # compute atoms activation intensities
    n_splits, n_channels, n_times = X_split.shape
    X = X_split.swapaxes(0, 1).reshape(n_channels, n_times * n_splits)
    z_hat = cdl.transform(X[None, :])

    dict_res = dict(
        cdl_params=cdl_params,
        u_hat_=u_hat_,
        v_hat_=v_hat_,
        z_hat=z_hat,
        events=info["temp"]["events"],
        event_id=info["temp"]["event_id"],
        sfreq=sfreq,
        info=info,
        # Duration of the experiment, in seconds
        T=(n_times * n_splits) / sfreq,
    )

    return dict_res


def run_cdl_sample(
    sfreq=150.0,
    n_atoms=40,
    n_times_atom=150,
    reg=0.1,
    n_iter=100,
    eps=1e-4,
    n_jobs=5,
    n_splits=10,
):
    """Run Convolutional Dictionary Learning on mne.sample."""
    load_params = dict(sfreq=sfreq, n_splits=n_splits)
    return _run_cdl_data(
        dataset="sample",
        load_params=load_params,
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        n_iter=n_iter,
        eps=eps,
        n_jobs=n_jobs,
    )


def run_cdl_somato(
    sfreq=150.0,
    n_atoms=25,
    n_times_atom=75,
    reg=0.2,
    n_iter=100,
    eps=1e-4,
    use_greedy=False,
    n_jobs=5,
    n_splits=10,
):
    """Run Convolutional Dictionary Learning on mne.somato."""
    load_params = dict(sfreq=sfreq, n_splits=n_splits)
    return _run_cdl_data(
        dataset="somato",
        load_params=load_params,
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        n_iter=n_iter,
        eps=eps,
        n_jobs=n_jobs,
    )


def run_cdl_camcan(
    subject_id="CC320428",
    sfreq=150.0,
    n_atoms=30,
    n_times_atom=int(np.round(0.7 * 150.0)),
    reg=0.2,
    n_iter=100,
    eps=1e-5,
    tol_z=1e-3,
    use_greedy=False,
    n_jobs=5,
    n_splits=10,
):
    """Run Convolutional Dictionary Learning on Cam-CAN dataset.

    Parameters
    ----------
    subject_id : str
        For Cam-CAN dataset, the subject id to run the CSC on. Defaults to
        'CC620264', a 76.33 year-old woman.

    """
    load_params = dict(sfreq=sfreq, n_splits=n_splits)
    return _run_cdl_data(
        dataset="camcan",
        subject_id=subject_id,
        load_params=load_params,
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        n_iter=n_iter,
        eps=eps,
        tol_z=tol_z,
        use_greedy=use_greedy,
        n_jobs=n_jobs,
    )
