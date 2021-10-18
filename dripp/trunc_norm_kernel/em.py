"""
Utils functions used in the computation of the next EM iteration
"""

from re import M
import numpy as np
from scipy.stats import norm

from dripp.trunc_norm_kernel.utils import get_driver_delays


# EPS = np.finfo(float).eps
EPS = 1e-5


def compute_C(m, sigma, a, b):
    """

    """
    if not b > a:
        raise ValueError("truncation values must be sorted, with a < b")

    C = sigma * np.sqrt(2 * np.pi)
    C *= norm.cdf((b - m) / sigma) - norm.cdf((a - m) / sigma)

    return C


def compute_C_m(m, sigma, a, b):
    """

    """

    if not b > a:
        raise ValueError("truncation values must be sorted, with a < b")

    C_m = np.exp(- np.square(a-m) / (2 * np.square(sigma)))
    C_m -= np.exp(- np.square(b-m) / (2 * np.square(sigma)))

    return C_m


def compute_C_sigma(m, sigma, a, b):
    """

    """

    if not b > a:
        raise ValueError("truncation values must be sorted, with a < b")

    C_sigma = (a - m) / sigma * \
        np.exp(- np.square(a-m) / (2 * np.square(sigma)))
    C_sigma -= (b - m) / sigma * \
        np.exp(- np.square(b-m) / (2 * np.square(sigma)))
    C_sigma += compute_C(m, sigma, a, b) / sigma

    return C_sigma


def compute_Cs(kernel):
    """ Compute C coefficient and its derivatives C_m and C_sigma

    Parameters
    ----------
    kernel : array-like of model.TruncNormKernel objects

    Returns
    -------
    tuple of size 3
        (C, C_m, C_sigma)

    """
    C, C_m, C_sigma = [], [], []
    for this_kernel in kernel:
        m = this_kernel.m
        sigma = this_kernel.sigma
        lower, upper = this_kernel.lower, this_kernel.upper

        C.append(compute_C(m, sigma, lower, upper))
        C_m.append(compute_C_m(m, sigma, lower, upper))
        C_sigma.append(compute_C_sigma(m, sigma, lower, upper))

    return C, C_m, C_sigma


def compute_p_tk(t, intensity, driver_delays=None):
    r"""Compute the probability that the activation at time $t$ has been
    triggered by the baseline intensity of the process $k$.

    .. math::

        P_{t,k} = \frac{\mu_k}{\lambda_{k,p}(t)}


    Parameters
    ----------
    t : float | array-like
        the time(s) we would like to compute p_tk at

    intensity : instance of model.Intensity
        intensity function

    last_tt : float | array-like
        the corresponding driver's timestamps

    Returns
    -------
    float | numpy.array

    """

    return intensity.baseline / intensity(t, driver_delays=driver_delays)


def compute_p_tp(t, intensity, driver_delays=None):
    r"""Compute the probability that the activation at time $t$ has been
    triggered by the driver $p$.

    .. math::

        P_{t,p} = \frac{\alpha_{k,p}\kappa_{k,p}\left(t - t_*^{(p)}(t)\right)}\
            {\lambda_{k,p}(t)}\mathds{1}_{t \geq t_1^{(p)}}

    Parameters
    ----------
    t : float | array-like
        the time(s) we would like to compute p_tp at

    intensity : instance of model.Intensity
        intensity function

    last_tt : float | array-like
        the corresponding driver's timestamps

    Returns
    -------
    float | numpy.array

    """

    t = np.atleast_1d(t)

    if driver_delays is None:
        driver_delays = get_driver_delays(intensity, t)
    # else:
    #     driver_delays = np.atleast_2d(driver_delays)

    # compute value of intensity function at each time t
    intensity_at_t = intensity(t, driver_delays=driver_delays)
    # for every driver, compute the associated P_tp
    list_p_tp = []
    for p, delays in enumerate(driver_delays):
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
        val = delays.copy()
        val.data = kernel(val.data)
        # p_tp = alpha * np.nansum(kernel(delays), axis=1)
        p_tp = alpha * val.sum(axis=1)
        p_tp /= intensity_at_t
        list_p_tp.append(p_tp)

    return np.array(list_p_tp)


def compute_next_baseline(intensity, T):
    r"""Compute the value of mu at the next EM step

    .. math::

        \mu_k^{(n+1)} = \frac{1}{T} \sum_{t\in\mathcal{A}_k} P_{t,k}^{(n)}

    Parameters
    ----------
    intensity : instance of model.Intensity
        intensity function

    T : int | float
        total duration of the process, in seconds

    Returns
    -------
    float

    """

    if intensity.baseline == 0:
        return 0

    sum_p_tk = np.nansum(compute_p_tk(intensity.acti_tt, intensity,
                                      driver_delays=intensity.driver_delays))
    return sum_p_tk / T


def compute_next_alpha_m_sigma(intensity, C, C_m, C_sigma):
    """Compute next values of parameters alpha, m and sigma

    Parameters
    ----------
    intensity : instance of model.Intensity
        intensity function

    C, C_m, C_sigma : float
        values of constants

    Returns
    -------
    tuple of 3 floats
        next value for alpha, m, sigma

    """

    # sum over all activation timestamps
    sum_p_tp = np.nansum(compute_p_tp(intensity.acti_tt, intensity,
                                      driver_delays=intensity.driver_delays),
                         axis=1)

    # new value of alpha
    n_drivers_tt = np.array(
        [this_driver_tt.size for this_driver_tt in intensity.driver_tt])

    # project on R+ is eventually done after
    next_alpha = sum_p_tp / n_drivers_tt

    # n_drivers = len(intensity.driver_tt)
    next_m, next_sigma = [], []
    # for p in range(intensity.n_drivers):
    for p, diff in enumerate(intensity.driver_delays):
        if next_alpha[p] == 0:
            next_m.append(intensity.kernel[p].m)
            next_sigma.append(intensity.kernel[p].sigma)
        else:
            # shape: (n_acti_tt, n_drivers_p_tt)
            # diff = intensity.acti_tt[:, np.newaxis] - intensity.driver_tt[p]
            # next value of m for p-th driver
            if C[p] > 0:  # avoid division by 0
                # sum over the driver events
                val = diff.copy()
                val.data *= intensity.kernel[p](val.data)
                sum_temp_m = val.sum(axis=1)
                # sum_temp_m = np.nansum(diff * intensity.kernel[p](diff),
                #                        axis=1)
                sum_temp_m /= intensity(intensity.acti_tt,
                                        driver_delays=intensity.driver_delays)
                this_next_m = intensity.alpha[p] * sum_temp_m.sum() / \
                    sum_p_tp[p]
                this_next_m -= np.square(
                    intensity.kernel[p].sigma) * C_m[p] / C[p]
            else:
                this_next_m = intensity.kernel[p].m
            next_m.append(this_next_m)
            # next value of sigma for p-th driver
            if C_sigma[p] > 0:  # avoid division by 0
                # compute diff from the current kernel mean
                val = diff.copy()
                val.data = np.square(val.data - intensity.kernel[p].m) \
                    * intensity.kernel[p](val.data)
                sum_temp_sigma = val.sum(axis=1)
                # diff_m = diff - intensity.kernel[p].m
                # sum over the driver events
                # sum_temp_sigma = np.nansum(np.square(diff_m) *
                #                            intensity.kernel[p](diff),
                #                            axis=1)
                sum_temp_sigma *= intensity.alpha[p] / \
                    intensity(intensity.acti_tt,
                              driver_delays=intensity.driver_delays)
                # cubic root
                this_next_sigma = np.cbrt(
                    C[p] / C_sigma[p] * sum_temp_sigma.sum() / sum_p_tp[p])
                # project on semi-closed set [EPS, +infty) to stay in
                # constraint space
                this_next_sigma = max(this_next_sigma, EPS)
            else:
                this_next_sigma = intensity.kernel[p].sigma
            next_sigma.append(this_next_sigma)

    return next_alpha, next_m, next_sigma


def compute_nexts(intensity, T):
    """Compute next values for the 4 parameters

    Parameters
    ----------
    intensity : instance of model.Intensity
        intensity function

    T : int | float
        total duration of the process, in seconds


    Returns
    -------
    tuple of 4 floats
        next value for baseline, alpha, m, sigma

    """

    baseline = compute_next_baseline(intensity, T)
    C, C_m, C_sigma = compute_Cs(intensity.kernel)
    alpha, m, sigma = compute_next_alpha_m_sigma(intensity, C, C_m, C_sigma)

    return baseline, alpha, m, sigma
