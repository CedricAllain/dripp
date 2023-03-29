# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
import mne_bids

from alphacsc.datasets import somato
from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel
from dripp.trunc_norm_kernel.optim import em_truncated_norm, initialize
from dripp.experiments.run_cdl import run_cdl_somato
from dripp.trunc_norm_kernel.utils import get_last_timestamps

N_JOBS = 40  # number of jobs to run in parallel. To adjust based on machine

cdl_params = {
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'n_jobs': 5,
    'n_splits': 10,
    'n_atoms': 20,
    'n_times_atom': 80,
    'reg': 0.2
}
# run CDL and EM
lower, upper = 0, 2
shift_acti = True
threshold = 1e-10
n_iter = 400


# get CDL results
dict_global = run_cdl_somato(**cdl_params)

# get general parameters
list_atoms = list(range(cdl_params['n_atoms']))
list_tasks = [1]
dict_other_params = dict_global['dict_other_params']
sfreq = dict_other_params['sfreq']
dict_pair_up = dict_global['dict_pair_up']
T = dict_pair_up['T']

# get events timestamps
events_timestamps = dict_pair_up['events_timestamps']  # type = dict
# get activations and filter
acti = np.array(dict_pair_up['acti_shift'])
atom_to_filter = 'all'
time_interval = 0.01
acti = utils.filter_activation(
    acti, atom_to_filter, sfreq, time_interval)
# apply threshlold and get activations timestamps
atoms_timestamps = utils.get_atoms_timestamps(acti=acti,
                                              sfreq=sfreq,
                                              threshold=threshold)
# eyeblink artifact
kk = 10
acti_tt = atoms_timestamps[kk]
driver_tt = np.r_[events_timestamps[list_tasks[0]]]

# %%

# "old" parameters initialization


def initialize_alpha(baseline, ppt_in_support, ppt_of_support):
    """Initializa parameter alpha for the "smart start" initialization strategy

    Parameters
    ----------
    baseline : float
        intensity baseline parameter

    ppt_in_support : float
        proportion of activation that kend in kernel support

    ppt_of_support : float
        proportion of all kernel supports over T


    Returns
    -------
    float

    """

    if ppt_in_support == 1 or baseline == 0:
        return 1

    a = np.exp(baseline)
    lim = ((a-1) / 5 + 1 / (1 - ppt_of_support)) ** (-1) + ppt_of_support
    alpha_init = -a * np.log((lim - ppt_in_support) / (lim - ppt_of_support))

    return max(alpha_init, 0)  # project on [0 ; +infty]


EPS = np.finfo(float).eps


def old_initialize(acti_tt=(), driver_tt=(), lower=30e-3, upper=500e-3, T=60,
                   initializer='smart_start', seed=None):
    """Initializa EM 4 parameters (baseline, alpha, m and sigma) given an
    initialization method

    Parameters
    ----------

    acti_tt : array-like

    driver_tt : array-like

    lower, upper : float
        kernel's truncation values
        default is 30e-3, 500e-3

    T : int | float
        total duration
        default is 60

    initializer: 'random' | 'smart_start'
        method used to initialize parameters
        if 'random', initial values are draw from uniform distributions
        if 'smart_start', empirical values of m and sigma are computed and
        used as initial values.
        default is 'smart_start'

    seed : int | None
        used to set a numpy RandomState
        default is None

    Returns
    -------
    tuple of size 4
        initial values for baseline, alpha, m and sigma

    """

    acti_tt = np.array(acti_tt)
    driver_tt = np.array(driver_tt)

    if initializer == 'random':
        rng = np.random.RandomState(seed)
        baseline_init = rng.uniform(low=0.15, high=1)
        m_init = rng.uniform(low=max(lower, 0.1), high=upper)
        sigma_init = rng.uniform(low=5e-2, high=1)
        alpha_init = rng.uniform(low=0.15, high=1)

    elif initializer == 'smart_start':
        if acti_tt.size == 0:
            baseline_init = 0
            alpha_init = 0
            m_init, sigma_init = np.nan, EPS
            return baseline_init, alpha_init, m_init, sigma_init

        # set of all activations that lend in a kernel support
        diff = acti_tt - get_last_timestamps(driver_tt, acti_tt)[0]
        mask = (diff <= upper) * (diff >= lower)
        acti_in_support = acti_tt[mask]
        # initialize baseline
        baseline_init = acti_tt.size - acti_in_support.size
        baseline_init /= (T - driver_tt.size * (upper - lower))
        # initialize m and sigma
        delays = diff[mask]
        if delays.size == 0:
            alpha_init = 0
            m_init, sigma_init = np.nan, EPS
            return baseline_init, alpha_init, m_init, sigma_init

        m_init, sigma_init = np.mean(delays), np.std(delays)
        sigma_init = max(EPS, sigma_init)  # project on [EPS ; +infty]

        # proportion of activation that kend in kernel support
        ppt_in_support = acti_in_support.size / acti_tt.size
        # proportion of all supports over T
        ppt_of_support = driver_tt.size * (upper - lower) / T
        # initializa alpha
        alpha_init = initialize_alpha(baseline=baseline_init,
                                      ppt_in_support=ppt_in_support,
                                      ppt_of_support=ppt_of_support)
    else:
        raise ValueError("Initializer method %s is unknown" % initializer)

    return baseline_init, alpha_init, m_init, sigma_init


init_params = old_initialize(acti_tt=acti_tt, driver_tt=driver_tt, lower=lower,
                             upper=upper, T=T, initializer='smart_start')
baseline_hat, alpha_hat, m_hat, sigma_hat = init_params
init_params = baseline_hat, [alpha_hat], [m_hat], [sigma_hat]

print(init_params)
# %%

new_init = initialize(acti_tt=acti_tt, driver_tt=driver_tt,
                      lower=lower, upper=upper, T=T, initializer='smart_start')
print(new_init)

# %%

res_em = em_truncated_norm(
    acti_tt=acti_tt,
    driver_tt=driver_tt,
    lower=lower,
    upper=upper,
    T=T,
    init_params=init_params,
    early_stopping=None,
    early_stopping_params={},
    alpha_pos=True,
    n_iter=n_iter,
    verbose=True,
    disable_tqdm=False)

# %%
