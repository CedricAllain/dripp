import numpy as np
from functools import partial
from tqdm import tqdm

from .utils import get_last_timestamps
from .model import TruncNormKernel, Intensity
from .metric import negative_log_likelihood
from .em import compute_nexts, compute_Cs


EPS = np.finfo(float).eps


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


def initialize(acti_tt=(), driver_tt=(), lower=30e-3, upper=500e-3, T=60,
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
        diff = acti_tt - get_last_timestamps(driver_tt, acti_tt)
        mask = (diff < upper) * (diff > lower)
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


def compute_baseline_mle(acti_tt=(), T=60, return_nll=True):
    r"""Compute the Maximum Liklihood Estimator (MLE) of the baseline, and the
    corresponding negative log-likehood (nll).

    .. math::
        \mu_k^{(MLE)} = \frac{\#\mathcal{A}_k}{T}
        \mathcal{L}_{k,p} = \mu_k^{(MLE)} T - \#\mathcal{A}_k \log\mu_k^{(MLE)}

    Parameters
    ----------
    acti_tt : array-like
        process's activation timestamps

    T : int | float

    return_nll : bool
        if True, compute and return the corresponding negative log-likehood

    Returns
    -------
    if return_nll :
        tuple of size 2
            baseline MLE and the corresponding negative log-likehood
    else:
        float
            baseline MLE

    """
    acti_tt = np.array(acti_tt)

    baseline_mle = acti_tt.size / T

    if return_nll:
        nll = T * baseline_mle - acti_tt.size * np.log(baseline_mle)
        return baseline_mle, nll
    else:
        return baseline_mle


def early_stopping_percent_mass(kernel, p_mass=0.99, n_tt=4,
                                sfreq=150.):
    """ If `p_mass` % of kernel's mass is concentraded on less than `n_tt`
    timestamps, the EM algorithm stops.

    Parameters
    ----------
    kernel : model.Kernel object

    p_mass : float
        percentage of mass to be concentrated in less that `n_tt` timestamps in
        order to stop the algorithm

    n_tt : int
        number of timestamps

    sfreq : float
        sampling frequency
        Default is 150.

    Returns
    -------
    bool

    """

    # determine quantiles of interest
    # q1, q2 = kernel.ppf((1 - p_mass) / 2), kernel.ppf((1 + p_mass) / 2)
    q1, q2 = kernel.interval(p_mass)
    # if the inter quantile intervalle is too small, it means the mass is too
    # concentrate, i.e., sigma is too small to mean anything
    if (q1 == q2) or (np.abs(q2 - q1) < n_tt / sfreq):
        return True

    return False


def stop_em(alpha, kernel, early_stopping=None, verbose=True,
            **early_stopping_params):
    """ Determine if the EM algo needs to be stopped.
    Returns True if alpha = 0, or if sigma is too small to be meaningfull.

    Parameters
    ----------
    alpha : float | int
        value of the alpha parameter

    kernel : instance of TruncNormKernel
        kernel function

    early_stopping : "early_stopping_percent_mass" | None
        method to deal with pathological cases
        default is None

    verbose : bool
        if True, print some informations

    early_stopping_params : dict
        criteria for `early_stopping` function, if not None

    Returns
    -------
    bool : either or not to stop the EM algo

    """

    if alpha == 0:
        if verbose:
            print("Earling stopping due to alpha = 0.")

        return True

    stop_sigma = False

    if early_stopping == "early_stopping_percent_mass":
        stop_sigma = early_stopping_percent_mass(
            kernel, **early_stopping_params)

    C, _, C_sigma = compute_Cs(kernel)
    stop_C = (C == 0)  # or (C_sigma <= 0)

    if stop_sigma or stop_C:
        if verbose:
            print("Earling stopping: kernel's mass is too concentrated, "
                  "either sigma is too small to continue or m is too far "
                  "from kernel's support. "
                  "The process is hence considered not driven")
        return True

    return False


def em_truncated_norm(acti_tt, driver_tt=(),
                      lower=30e-3, upper=500e-3, T=60, sfreq=150.,
                      init_params=None, initializer='smart_start',
                      early_stopping=None, early_stopping_params={},
                      alpha_pos=True, n_iter=80,
                      verbose=False, disable_tqdm=False):
    """Run EM-based algorithm

    Parameters
    ----------
    acti_tt : array-like

    driver_tt : array-like

    lower, upper : float
        kernel's truncation values

    T : int | float
        total duration of the process, in seconds

    sfreq : int | None
        sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    init_params: tuple | None
        intial values of (baseline, alpha, m, sigma)
        if None, intialize with initializer method
        default is None

    initializer: 'random' | 'smart_start'
        method to initalize parameters
        default is 'smart_start'

    early_stopping : string
        "early_stopping_sigma" | "early_stopping_percent_mass" | None
        method used for early stopping
        default is None

    early_stopping_params : dict
        parameters for the early stopping method
        for 'early_stopping_sigma', keys must be 'n_sigma', 'n_tt' and 'sfreq'
        for 'early_stopping_percent_mass', keys must be 'alpha', 'n_tt' and
        'sfreq'

    alpha_pos : bool
        if True, force alpha to be non-negative

    n_iter : int
        number of iterations
        default is 80

    verbose : bool
        if True, will print some informations
        default is False

    disable_tqdm : bool
        if True, will print a progress bar
        default is False


    Returns
    -------
    res_params : tuple of size 4
        values of learned parameters baseline, alpha, m and sigma

    history_params : tuple of 4 1d-numpy.array
        for every learned parameter, its history over all EM iterations

    hist_loss : 1d numpy.array
        value of the negative log-likelihood over all EM iterations
    """

    acti_tt = np.array(acti_tt)
    driver_tt = np.array(driver_tt)

    assert acti_tt.size > 0, "no activation vector was given"

    if driver_tt.size == 0:
        if verbose:
            print("Intensity has no driver timestamps. "
                  "Will return baseline MLE and corresponding loss "
                  "(negative log-likelihood).")
        return compute_baseline_mle(acti_tt, T)

    # initialize parameters
    if init_params is None:
        init_params = initialize(
            acti_tt, driver_tt, lower, upper, T, initializer=initializer)
        if verbose:
            print("Initials parameters:\n(mu, alpha, m, sigma) = ",
                  init_params)

    baseline_hat, alpha_hat, m_hat, sigma_hat = init_params

    # initialize kernel, intensity functions as well as nll function
    kernel = TruncNormKernel(lower, upper, m_hat, sigma_hat, sfreq=sfreq)
    intensity = Intensity(baseline_hat, alpha_hat, kernel, driver_tt, acti_tt)
    nll = partial(negative_log_likelihood, T=T)

    # initializa history of parameters and loss
    hist_baseline, hist_alpha = [baseline_hat], [alpha_hat]
    hist_m, hist_sigma = [m_hat], [sigma_hat]
    hist_loss = [nll(intensity)]

    if verbose:
        print("Initial loss (negative log-likelihood):", hist_loss[0])

    for n in tqdm(range(int(n_iter)), disable=disable_tqdm):
        stop = stop_em(alpha_hat, kernel,
                       early_stopping, verbose,
                       **early_stopping_params)
        if stop:  # either alpha = 0, or mass is too concentrated
            alpha_hat == 0
            baseline_hat = compute_baseline_mle(acti_tt, T, return_nll=False)
            break
        # compute next values of parameters
        nexts = compute_nexts(intensity, T)
        baseline_hat, alpha_hat, m_hat, sigma_hat = nexts
        # force alpha to stay non-negative
        if alpha_pos and (alpha_hat < 0):
            if verbose:
                print("alpha_hat was negative, alpha_hat is thus set to 0.")
            alpha_hat = 0
            baseline_hat = compute_baseline_mle(acti_tt, T, return_nll=False)
            break

        # append history
        hist_baseline.append(baseline_hat)
        hist_alpha.append(alpha_hat)
        hist_m.append(m_hat)
        hist_sigma.append(sigma_hat)
        # update kernel function
        kernel.update(m=m_hat, sigma=sigma_hat)
        # update intensity function
        intensity.baseline = baseline_hat
        intensity.alpha = alpha_hat
        intensity.kernel = kernel
        # compute loss
        hist_loss.append(nll(intensity))

    res_params = baseline_hat, alpha_hat, m_hat, sigma_hat

    if verbose:
        print("Optimal parameters:\n(mu, alpha, m, sigma) = ", res_params)

    history_params = hist_baseline, hist_alpha, hist_m, hist_sigma

    return res_params, np.array(history_params), np.array(hist_loss)
