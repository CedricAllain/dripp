"""Functions that are called for parameters initialisation and EM
computation.
"""
# %%
import numpy as np
import time
import numbers
from functools import partial
from scipy.sparse import find
from tqdm import tqdm

from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood
from dripp.trunc_norm_kernel.em import compute_nexts
from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.utils import check_acti_tt, check_driver_tt

from dripp.utils import profile_this

EPS = np.finfo(float).eps


def compute_lebesgue_support(all_tt, lower, upper):
    """Compute the Lebesgue measure of the union of the kernels supports
    following a set of timestamps.

    Compute lebesgue_measure(Union{[tt + lower, tt + upper] for tt in all_tt})

    Parameters
    ----------
    all_tt : array-like
        The set of all timestamps that induce a kernel support.

    lower, upper : float
        Lower and upper bounds of the truncated gaussian kernel.

    Returns
    -------
    float
        The Lesbegue measure of the supports union.
    """
    s = 0
    temp = (all_tt[0] + lower, all_tt[0] + upper)
    for i in range(all_tt.size - 1):
        if all_tt[i+1] + lower > temp[1]:
            s += temp[1] - temp[0]
            temp = (all_tt[i+1] + lower, all_tt[i+1] + upper)
        else:
            temp = (temp[0], all_tt[i+1] + upper)

    s += temp[1] - temp[0]
    return s


def initialize_baseline(intensity, T=None):
    """ Initialize the baseline parameter with a smart strategy.

    The initial value correspond of the average number of activations that lend
    outside any kernel support.

    Parameters
    ----------
    intensity : instance of model.Intensity
        The intensity object that contains the different drivers.

    T : int | float | None
        Duration of the process. If None, is set to the maximum the intensity
        activation timestamps plus a margin equal to the upper truncation
        value. Defaults to None.

    Returns
    -------
    float
        The initial value of the the baseline parameter with a smart strategy.
    """
    # compute the number of activation that lend in at least one kernel's
    # support
    acti_in_support = []
    for delays in intensity.driver_delays:
        # get the colons (i.e., the activation tt) for wich there is at least
        # one "good" delay)
        acti_in_support.extend(find(delays)[0])

    # compute the Lebesgue measure of all kernels' supports
    all_tt = np.sort(np.hstack(intensity.driver_tt))
    lower, upper = intensity.kernel[0].lower, intensity.kernel[0].upper
    s = compute_lebesgue_support(all_tt, lower, upper)
    if T is None:
        T = intensity.acti_tt.max() + upper
    baseline_init = (len(intensity.acti_tt) -
                     len(set(acti_in_support))) / (T - s)
    return baseline_init


def initialize(intensity, T=None, initializer='smart_start', seed=None):
    """Initializa EM 4 parameters (baseline, alpha, m and sigma) given an
    initialization method.

    Parameters
    ----------
    intensity : instance of model.Intensity

    T : int | float | None
        Duration of the process. Defaults to None.

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
        alpha, m and sigma are array-like of shape (n_drivers, )
    """
    driver_tt = intensity.driver_tt
    n_drivers = len(driver_tt)

    lower, upper = intensity.kernel[0].lower, intensity.kernel[0].upper

    if initializer == 'random':
        rng = np.random.RandomState(seed)
        baseline_init = rng.uniform(low=0.15, high=1)
        m_init = rng.uniform(low=max(lower, 0.1), high=upper, size=n_drivers)
        sigma_init = rng.uniform(low=5e-2, high=1, size=n_drivers)
        alpha_init = rng.uniform(low=0.15, high=1, size=n_drivers)

    elif initializer == 'smart_start':
        # default values
        default_m = (upper - lower) / 2
        default_sigma = 0.95 * (upper - lower) / 4

        if intensity.acti_tt.size == 0:   # no activation at all on the process
            baseline_init = 0
            alpha_init = np.full(n_drivers, fill_value=0)
            m_init = np.full(n_drivers, fill_value=default_m)
            sigma_init = np.full(n_drivers, fill_value=default_sigma)
            return baseline_init, alpha_init, m_init, sigma_init

        # initialize baseline
        baseline_init = initialize_baseline(intensity, T)

        alpha_init = []
        m_init = []
        sigma_init = []
        for p, delays in enumerate(intensity.driver_delays):
            delays = delays.data
            if delays.size == 0:
                alpha_init.append(- baseline_init)
                m_init.append(default_m)
                sigma_init.append(default_sigma)
            else:
                # compute Lebesgue measure of driver p supports
                s = compute_lebesgue_support(driver_tt[p], lower, upper)
                alpha_init.append(delays.size / s - baseline_init)
                # alpha_init.append(
                #     (intensity.acti_tt.size - baseline_init * T) / s)
                m_init.append(np.mean(delays))
                sigma_init.append(max(EPS, np.std(delays)))
    else:
        raise ValueError("Initializer method %s is unknown" % initializer)

    return baseline_init, alpha_init, m_init, sigma_init


def compute_baseline_mle(acti_tt, T=None, return_nll=True):
    r"""Compute the Maximum Liklihood Estimator (MLE) of the baseline, and the
    corresponding negative log-likehood (nll).

    .. math::
        \mu_k^{(MLE)} = \frac{\#\mathcal{A}_k}{T}
        \mathcal{L}_{k,p} = \mu_k^{(MLE)} T - \#\mathcal{A}_k \log\mu_k^{(MLE)}

    Parameters
    ----------
    acti_tt : array-like
        Process's activation timestamps.

    T : int | float | None
        Duration of the process. If None, is set to the maximum the intensity
        activation timestamps plus a margin equal to the upper truncation
        value. Defaults to None.

    return_nll : bool
        If True, compute and return the corresponding negative log-likehood.
        Defaults to True.

    Returns
    -------
    if return_nll :
        tuple of size 2
            baseline MLE and the corresponding negative log-likehood
    else:
        float
            baseline MLE
    """
    acti_tt = check_acti_tt(acti_tt)
    if T is None:
        T = intensity.acti_tt.max() + intensity.kernel[0].upper

    baseline_mle = acti_tt.size / T

    if return_nll:
        nll = T * baseline_mle - acti_tt.size * np.log(baseline_mle)
        return baseline_mle, nll
    else:
        return baseline_mle


# @profile_this
def em_truncated_norm(acti_tt, driver_tt=None,
                      lower=30e-3, upper=500e-3, T=None, sfreq=150.,
                      init_params=None, initializer='smart_start',
                      alpha_pos=True, n_iter=80,
                      verbose=False, disable_tqdm=False, compute_loss=False):
    """Run EM-based algorithm.

    Parameters
    ----------
    acti_tt : array-like
        Process's activation timestamps.

    driver_tt : list of arrays | array
        List of length n_drivers. Each element contains the events
        of one driver. Defaults to None.

    lower, upper : float
        Kernel's truncation values. Defaults to lower = 30e-3 and
        upper = 500e-3.

    T : int | float | None
        Duration of the process. If None, is set to the maximum the intensity
        activation timestamps plus a margin equal to the upper truncation
        value. Defaults to None.

    sfreq : int | None
        Sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values. If None, the kernel will be
        exactly evaluate at each call. Warning: setting sfreq to None may
        considerably increase computational time. Defaults to 150.

    init_params: tuple | None
        Intial values of (baseline, alpha, m, sigma). If None, intialize with
        initializer method. Defaults to None.

    initializer: 'random' | 'smart_start'
        Method to initalize parameters. Defaults to 'smart_start'.

    alpha_pos : bool
        If True, force alpha to be non-negative. Defaults to True.

    n_iter : int
        Number of iterations. Defaults to 80.

    verbose : bool
        If True, will print some informations. Defaults to False.

    disable_tqdm : bool
        If True, will print a progress bar. Defaults to False.

    compute_loss : bool
        If True, compute the initial and final loss values, as well as the loss
        at each EM iteration, and return the history of loss during the EM.
        Defaults to False.

    Returns
    -------
    res_params : tuple of size 4
        Values of learned parameters baseline, alpha, m and sigma.

    history_params : dict of array-like
        For every learned parameter, its history over all EM iterations.

    hist_loss : 1d numpy.array
        Value of the negative log-likelihood over all EM iterations.
    """

    acti_tt = check_acti_tt(acti_tt)
    assert acti_tt.size > 0, "No activation vector was given"

    driver_tt = check_driver_tt(driver_tt)
    n_drivers = len(driver_tt)

    if T is None:
        T = acti_tt.max() + upper

    if len(driver_tt) == 0:
        if verbose:
            print("Intensity has no driver timestamps. "
                  "Will return baseline MLE and corresponding loss "
                  "(negative log-likelihood).")
        return compute_baseline_mle(acti_tt, T)

    # define intances of kernels and intensity function
    kernel = [TruncNormKernel(lower, upper, sfreq=sfreq)] * n_drivers
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)

    # initialize parameters
    if init_params is None:
        init_params = initialize(intensity, T, initializer=initializer)
        if verbose:
            print("Initials parameters:\n(mu, alpha, m, sigma) = ",
                  init_params)

    baseline_hat, alpha_hat, m_hat, sigma_hat = init_params
    if alpha_pos:
        alpha_hat = np.array(alpha_hat).clip(min=0)
    # update kernels and intensity function
    intensity.update(baseline_hat, alpha_hat, m_hat, sigma_hat)

    # initialize history of parameters and loss
    history_params = {'baseline': [baseline_hat],
                      'alpha': [alpha_hat],
                      'm': [m_hat],
                      'sigma': [sigma_hat]}
    if compute_loss:
        # define loss function
        nll = partial(negative_log_likelihood, T=T)
        hist_loss = [nll(intensity)]
    else:
        hist_loss = []

    if verbose and compute_loss:
        print("Initial loss (negative log-likelihood):", hist_loss[0])

    stop = False
    for n in tqdm(range(int(n_iter)), disable=disable_tqdm):
        # compute next values of parameters
        baseline_hat, alpha_hat, m_hat, sigma_hat = compute_nexts(intensity, T)
        if alpha_pos:
            # force alpha to stay non-negative
            alpha_hat = alpha_hat.clip(min=0)  # project on R+
            if(alpha_hat.max() == 0):  # all alphas are zero
                if verbose:
                    print("alpha is null, compute baseline MLE.")
                baseline_hat, nll_mle = compute_baseline_mle(
                    acti_tt, T, return_nll=True)
                stop = True

        # append history
        history_params['baseline'].append(baseline_hat)
        history_params['alpha'].append(alpha_hat)
        history_params['m'].append(m_hat)
        history_params['sigma'].append(sigma_hat)

        if stop:
            if compute_loss:
                hist_loss.append(nll_mle)
            break
        else:
            intensity.update(baseline_hat, alpha_hat, m_hat, sigma_hat)
            # compute loss
            if compute_loss:
                hist_loss.append(nll(intensity))
    res_params = baseline_hat, alpha_hat, m_hat, sigma_hat

    if verbose:
        print("Optimal parameters:\n(mu, alpha, m, sigma) = ", res_params)

    return res_params, history_params, np.array(hist_loss)


if __name__ == '__main__':
    N_DRIVERS = 2
    T = 10_000  # process time, in seconds
    lower, upper = 30e-3, 800e-3
    sfreq = 500.
    start_time = time.time()
    driver_tt, acti_tt, _, intensity = simulate_data(
        lower=lower, upper=upper,
        m=[400e-3, 400e-3], sigma=[0.2, 0.05],
        sfreq=sfreq,
        baseline=0.8, alpha=[-0.8, 0.8],
        T=T, isi=[1, 1.2], n_tasks=0.8,
        n_drivers=N_DRIVERS, seed=0, return_nll=False, verbose=False)
    simu_time = time.time() - start_time
    print("Simulation time for %i driver(s) over %i seconds: %.3f seconds"
          % (N_DRIVERS, T, simu_time))

    start_time = time.time()
    res_params, history_params, _ = em_truncated_norm(
        acti_tt, driver_tt, lower, upper, T, sfreq, alpha_pos=False,
        n_iter=100, verbose=True)
    em_time = time.time() - start_time
    print('EM time', em_time)
    print("baseline_hat, alpha_hat, m_hat, sigma_hat:\n", res_params)

# %%