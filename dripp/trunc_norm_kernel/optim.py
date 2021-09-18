"""
XXX
"""
# %%
import numpy as np
from functools import partial
from numpy.core.records import array
from tqdm import tqdm

from dripp.trunc_norm_kernel.utils import get_last_timestamps
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood
from dripp.trunc_norm_kernel.em import compute_nexts, compute_Cs
from dripp.trunc_norm_kernel.simu import simulate_data


EPS = np.finfo(float).eps


def initialize_baseline(acti_tt=(), driver_tt=(), lower=30e-3, upper=500e-3,
                        T=60):
    """

    """
    # compute the number of activation that lend in at least one kernel's support
    acti_in_support = set()
    for p in range(driver_tt.shape[0]):
        # (n_driver_tt, n_acti_tt)
        delays = acti_tt - driver_tt[p][:, np.newaxis]
        mask = (delays >= lower) & (delays <= upper)
        acti_in_support.update(*[set(acti_tt[this_mask])
                                 for this_mask in mask])

    # compute the Lebesgue measure of all kernels' supports
    all_tt = np.sort(np.hstack(driver_tt.flatten()))
    s = 0
    temp = (all_tt[0] + lower, all_tt[0] + upper)
    for i in range(all_tt.size - 1):
        if all_tt[i+1] + lower > temp[1]:
            s += temp[1] - temp[0]
            temp = (all_tt[i+1] + lower, all_tt[i+1] + upper)
        else:
            temp = (temp[0], all_tt[i+1] + upper)

    s += temp[1] - temp[0]

    baseline_init = (acti_tt.size - len(acti_in_support)) / (T - s)
    return baseline_init


def initialize_alpha(baseline, ppt_in_support, ppt_of_support):
    """Initializa parameter alpha for the "smart start" initialization strategy

    Parameters
    ----------
    baseline : float
        intensity baseline parameter

    ppt_in_support : float | array-like
        proportion of activation that kend in kernel support

    ppt_of_support : float | array-like
        proportion of all kernel supports over T


    Returns
    -------
    array-like

    """

    ppt_in_support_list = np.atleast_1d(ppt_in_support)
    ppt_of_support_list = np.atleast_1d(ppt_of_support)

    assert len(ppt_in_support_list) == len(ppt_of_support_list)

    alpha_init_list = []
    for ppt_in_support, ppt_of_support in zip(ppt_in_support_list,
                                              ppt_of_support_list):

        if ppt_in_support == 1 or baseline == 0:
            alpha_init_list.append(1)
            continue

        a = np.exp(baseline)
        lim = ((a-1) / 5 + 1 / (1 - ppt_of_support)) ** (-1) + ppt_of_support
        alpha_init = -a * \
            np.log((lim - ppt_in_support) / (lim - ppt_of_support))

        alpha_init_list.append(max(alpha_init, 0))  # project on [0 ; +infty]

    return np.array(alpha_init_list)


def initialize(acti_tt=(), driver_tt=(), lower=30e-3, upper=500e-3, T=60,
               initializer='smart_start', seed=None):
    """Initializa EM 4 parameters (baseline, alpha, m and sigma) given an
    initialization method

    Parameters
    ----------

    acti_tt : array-like

    driver_tt : array-like

    lower, upper : float | array-like
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
        alpha, m and sigma are array-like of shape (n_drivers, )

    """

    acti_tt = np.atleast_1d(acti_tt)
    if isinstance(driver_tt[0], (int, float)):
        driver_tt = np.atleast_2d(driver_tt)
    driver_tt = np.array([np.array(x) for x in driver_tt], dtype=object)
    n_driver = driver_tt.shape[0]

    if initializer == 'random':
        rng = np.random.RandomState(seed)
        baseline_init = rng.uniform(low=0.15, high=1)
        m_init = rng.uniform(low=max(lower, 0.1), high=upper, size=n_driver)
        sigma_init = rng.uniform(low=5e-2, high=1, size=n_driver)
        alpha_init = rng.uniform(low=0.15, high=1, size=n_driver)

    elif initializer == 'smart_start':
        # default values
        default_alpha = 0.05
        default_m = (upper - lower) / 2
        default_sigma = 0.95 * (upper - lower) / 4

        if acti_tt.size == 0:  # no activation at all on the process
            baseline_init = 0
            alpha_init = np.full(n_driver, fill_value=0)
            m_init = np.full(n_driver, fill_value=default_m)
            sigma_init = np.full(n_driver, fill_value=default_sigma)
            return baseline_init, alpha_init, m_init, sigma_init

        # ------ initialize baseline ------
        baseline_init = initialize_baseline(
            acti_tt, driver_tt, lower, upper, T)

        # set of all activations that lend in a kernel support
        diff = acti_tt - get_last_timestamps(driver_tt, acti_tt)
        diff[np.isnan(diff)] = -1  # replace nan values
        mask = (diff <= upper) * (diff >= lower)

        alpha_init = []
        m_init = []
        sigma_init = []
        for p in range(n_driver):
            delays = diff[p][mask[p]]
            if delays.size == 0:
                alpha_init.append(default_alpha)
                m_init.append(default_m)
                sigma_init.append(default_sigma)
            else:
                alpha_init.append(delays.size / len(driver_tt[p]))
                m_init.append(np.mean(delays))
                sigma_init.append(max(EPS, np.std(delays)))

        # if n_driver == 1:
        #     acti_in_support = acti_tt[mask]
        # else:
        #     temp = []
        #     for mask_row in mask:
        #         temp.append(acti_tt[mask_row])
        #     acti_in_support = np.array(temp, dtype=object)
        # acti_tt_tiled = np.tile(acti_tt, (n_driver, 1))
        # acti_in_support = acti_tt_tiled[mask]
        # ------ initialize baseline ------
        # baseline_init = acti_tt.size
        # sum_support = 0
        # for aa, tt in zip(acti_in_support, driver_tt):
        #     baseline_init -= len(aa)
        #     sum_support += len(tt) * (upper - lower)
        # baseline_init /= (T - sum_support)
        # ------ initialize m and sigma ------
        # m_init = []
        # sigma_init = []
        # for i in range(n_driver):
        #     delays = diff[i][mask[i]]
        #     if delays.size == 0:
        #         m_init.append(np.nan)
        #         sigma_init.append(EPS)
        #         continue
        #         # return baseline_init, alpha_init, m_init, sigma_init

        #     m_init.append(np.mean(delays))
        #     # project on [EPS ; +infty]
        #     sigma_init.append(max(EPS, np.std(delays)))
        # m_init = np.array(m_init)
        # sigma_init = np.array(sigma_init)
        # # ----- initializa alpha ------
        # # proportion of activation that kend in kernel support
        # ppt_in_support = [len(aa) / len(acti_tt) for aa in acti_in_support]
        # # proportion of all supports over T
        # ppt_of_support = [len(tt) * (upper - lower) / T for tt in driver_tt]
        # # initializa alpha
        # alpha_init = initialize_alpha(baseline=baseline_init,
        #                               ppt_in_support=ppt_in_support,
        #                               ppt_of_support=ppt_of_support)
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
    stop_C = (np.array(C).max() == 0)  # or (C_sigma <= 0)

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
                      verbose=False, disable_tqdm=False, compute_loss=False):
    """Run EM-based algorithm

    Parameters
    ----------
    acti_tt : array-like

    driver_tt : array-like of shape (n_drivers, )

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

    compute_loss : bool
        if True, compute the initial and final loss values, as well as the loss
        at each EM iteration, and return the history of loss during the EM

    Returns
    -------
    res_params : tuple of size 4
        values of learned parameters baseline, alpha, m and sigma

    history_params : tuple of 4 1d-numpy.array
        for every learned parameter, its history over all EM iterations

    hist_loss : 1d numpy.array
        value of the negative log-likelihood over all EM iterations
    """

    acti_tt = np.atleast_1d(acti_tt)
    assert acti_tt.size > 0, "no activation vector was given"

    if np.array(driver_tt).size == 0:
        if verbose:
            print("Intensity has no driver timestamps. "
                  "Will return baseline MLE and corresponding loss "
                  "(negative log-likelihood).")
        return compute_baseline_mle(acti_tt, T)

    if isinstance(driver_tt[0], (int, float)):
        # if np.issubdtype(driver_tt[0].dtype, np.number):
        driver_tt = np.atleast_2d(driver_tt)
    driver_tt = np.array([np.array(x) for x in driver_tt])
    n_driver = driver_tt.shape[0]

    # initialize parameters
    if init_params is None:
        init_params = initialize(
            acti_tt, driver_tt, lower, upper, T, initializer=initializer)
        if verbose:
            print("Initials parameters:\n(mu, alpha, m, sigma) = ",
                  init_params)

    baseline_hat, alpha_hat, m_hat, sigma_hat = init_params

    # initialize kernels and intensity functions
    kernel = []
    for i in range(n_driver):
        kernel.append(TruncNormKernel(
            lower, upper, m_hat[i], sigma_hat[i], sfreq=sfreq))
    intensity = Intensity(baseline_hat, alpha_hat, kernel, driver_tt, acti_tt)

    # initializa history of parameters and loss
    hist_baseline, hist_alpha = [baseline_hat], [alpha_hat]
    hist_m, hist_sigma = [m_hat], [sigma_hat]
    if compute_loss:
        # define loss function
        nll = partial(negative_log_likelihood, T=T)
        hist_loss = [nll(intensity)]
    else:
        hist_loss = []

    if verbose:
        print("Initial loss (negative log-likelihood):", hist_loss[0])

    for n in tqdm(range(int(n_iter)), disable=disable_tqdm):
        # stop = stop_em(alpha_hat, kernel,
        #                early_stopping, verbose,
        #                **early_stopping_params)
        # if stop:  # either alpha = 0, or mass is too concentrated
        #     alpha_hat = np.full(n_driver, fill_value=0)
        #     baseline_hat = compute_baseline_mle(acti_tt, T, return_nll=False)
        #     break
        # compute next values of parameters
        nexts = compute_nexts(intensity, T)
        baseline_hat, alpha_hat, m_hat, sigma_hat = nexts
        # force alpha to stay non-negative
        if alpha_pos:
            alpha_hat = (alpha_hat).clip(min=0)  # project on R+
            if(alpha_hat.max() == 0):  # all alphas are zero
                if verbose:
                    print("alpha is null, compute baseline MLE.")
                # alpha_hat = np.full(n_driver, fill_value=0)
                baseline_hat = compute_baseline_mle(
                    acti_tt, T, return_nll=False)
                break

        # append history
        hist_baseline.append(baseline_hat)
        hist_alpha.append(alpha_hat)
        hist_m.append(m_hat)
        hist_sigma.append(sigma_hat)
        # # update kernel functions
        # for i in range(n_driver):
        #     kernel[i].update(m=m_hat[i], sigma=sigma_hat[i])
        # update intensity function
        # intensity.baseline = baseline_hat
        # intensity.alpha = alpha_hat
        # intensity.kernel = kernel
        intensity.update(baseline_hat, alpha_hat, m_hat, sigma_hat)
        # compute loss
        if compute_loss:
            hist_loss.append(nll(intensity))

    res_params = baseline_hat, alpha_hat, m_hat, sigma_hat

    if verbose:
        print("Optimal parameters:\n(mu, alpha, m, sigma) = ", res_params)

    history_params = hist_baseline, hist_alpha, hist_m, hist_sigma

    return res_params, np.array(history_params), np.array(hist_loss)


if __name__ == '__main__':
    N_DRIVERS = 2
    T = 2_000  # process time, in seconds
    lower, upper = 30e-3, 500e-3
    sfreq = 150.
    driver_tt, acti_tt, _, _ = simulate_data(
        lower=lower, upper=upper, m=150e-3, sigma=0.1, sfreq=sfreq,
        baseline=0.8, alpha=1, T=T, isi=1, n_tasks=0.4,
        n_drivers=N_DRIVERS, seed=None, return_nll=False, verbose=False)

    res_params = em_truncated_norm(acti_tt, driver_tt, lower, upper, T, sfreq,
                                   n_iter=300)[0]
    print("baseline_hat, alpha_hat, m_hat, sigma_hat:\n", res_params)

# %%
