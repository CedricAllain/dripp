"""
Utils functions used in the computation of the next EM iteration
"""

from re import M
import numpy as np
from scipy.stats import norm

from dripp.trunc_norm_kernel.utils import get_last_timestamps

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
    kernel : model.TruncNormKernel object

    Returns
    -------
    tuple of size 3
        (C, C_m, C_sigma)

    """
    m = kernel.m
    sigma = kernel.sigma
    lower, upper = kernel.lower, kernel.upper

    C = compute_C(m, sigma, lower, upper)
    C_m = compute_C_m(m, sigma, lower, upper)
    C_sigma = compute_C_sigma(m, sigma, lower, upper)

    return C, C_m, C_sigma


def compute_p_tk(t, intensity, last_tt=()):
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

    return intensity.baseline / intensity(t, last_tt=last_tt)


def compute_p_tp(t, intensity, last_tt=(), non_overlapping=False):
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

    list_p_tp = []
    if non_overlapping:
        last_tt = np.array(last_tt)

        if last_tt.size == 0:
            last_tt = get_last_timestamps(intensity.driver_tt, t)

        for alpha, kernel, this_last_tt in zip(intensity.alpha, intensity.kernel, last_tt):
            p_tp = alpha * kernel(t - this_last_tt)
            p_tp /= intensity(t, last_tt=last_tt)
            p_tp[np.isnan(p_tp)] = 0  # i.e., where kernel is not defined

            if t.size == 1:
                list_p_tp.append(p_tp[0])
            else:
                list_p_tp.append(p_tp)

    else:
        # lifting of the non-overlapping assumption
        for alpha, kernel in zip(intensity.alpha, intensity.kernel):

    return list_p_tp


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

    sum_p_tk = compute_p_tk(intensity.acti_tt, intensity,
                            last_tt=intensity.acti_last_tt).sum()
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

    p_tp = compute_p_tp(intensity.acti_tt, intensity,
                        last_tt=intensity.acti_last_tt)
    sum_p_tp = p_tp.sum()
    # if sum_p_tp == 0:
    #     print('sum_p_tp is null in compute_next_alpha_m_sigma')
    #     print('m, sigma = ', (intensity.kernel.m, intensity.kernel.sigma))
    #     import ipdb
    #     ipdb.set_trace()
    # new value of alpha
    next_alpha = sum_p_tp / intensity.driver_tt.size
    if next_alpha == 0:
        next_m = intensity.kernel.m
        next_sigma = intensity.kernel.sigma
    else:
        # new value of m
        diff = intensity.acti_tt - intensity.acti_last_tt
        sum_temp = np.nansum(diff * p_tp)
        next_m = sum_temp / sum_p_tp - \
            np.square(intensity.kernel.sigma) * C_m / C
        # new value of sigma
        sum_temp = np.nansum(np.square(diff - intensity.kernel.m) * p_tp)
        next_sigma = np.cbrt(C / C_sigma * sum_temp / sum_p_tp)  # cubic root
        # project on semi-closed set [EPS, +infty) to stay in constraint space
        next_sigma = max(next_sigma, EPS)

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
