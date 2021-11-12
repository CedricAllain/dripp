"""Utils functions used in the computation of the next EM iteration.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.utils import \
    get_driver_delays, check_truncation_values

EPS = 1e-5  # np.finfo(float).eps


def compute_C(m, sigma, a, b):
    """

    """
    check_truncation_values(a, b)

    C = sigma * np.sqrt(2 * np.pi)
    C *= norm.cdf((b - m) / sigma) - norm.cdf((a - m) / sigma)

    return C


def plot_C(m, sigma, a, b, save=False):

    def func(x):
        return np.exp(- (x-m)**2 / (2 * sigma**2))

    xx_min = min(np.floor(a), np.floor(m))
    xx_max = max(np.ceil(b), np.ceil(m))
    xx = np.linspace(xx_min, xx_max, 1000)
    plt.plot(xx, func(xx), label="exp(- (x-m)**2 / (2 * sigma**2)")
    plt.vlines(m, ymin=0, ymax=1, color='black',
               linestyle='--', label=r'm = %.3f' % m)
    xx_support = np.linspace(a, b, 600)
    plt.fill_between(xx_support, func(xx_support), step="pre", alpha=0.4,
                     label=r'C = %.3f' % compute_C(m, sigma, a, b))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(0, 1.1)
    plt.xlabel('x')
    plt.title(r'C(m=%.3f, sigma=%.3f, a=%.3f, b=%.3f)' % (m, sigma, a, b))
    plt.legend()
    if save:
        plt.savefig("constant_C.pdf")
    plt.show()


def compute_C_m(m, sigma, a, b):
    """

    """
    check_truncation_values(a, b)

    C_m = np.exp(- np.square(a-m) / (2 * np.square(sigma)))
    C_m -= np.exp(- np.square(b-m) / (2 * np.square(sigma)))

    return C_m


def compute_C_sigma(m, sigma, a, b):
    """

    """
    check_truncation_values(a, b)

    C_sigma = (a - m) / sigma * \
        np.exp(- np.square(a-m) / (2 * np.square(sigma)))
    C_sigma -= (b - m) / sigma * \
        np.exp(- np.square(b-m) / (2 * np.square(sigma)))
    C_sigma += compute_C(m, sigma, a, b) / sigma

    return C_sigma


def compute_Cs(kernel):
    """Compute C coefficient and its derivatives C_m and C_sigma.

    Parameters
    ----------
    kernel : array-like of model.TruncNormKernel objects

    Returns
    -------
    tuple of 3 lists
        (C, C_m, C_sigma)
    """
    C, C_m, C_sigma = [], [], []
    for this_kernel in kernel:
        m, sigma = this_kernel.m, this_kernel.sigma
        lower, upper = this_kernel.lower, this_kernel.upper
        C.append(compute_C(m, sigma, lower, upper))
        C_m.append(compute_C_m(m, sigma, lower, upper))
        C_sigma.append(compute_C_sigma(m, sigma, lower, upper))

    return C, C_m, C_sigma


def compute_p_tk(t, intensity, driver_delays=None):
    r"""Compute the probability that the activation at time $t$ has been
    triggered by the baseline intensity of the process $k$.

    .. math::

        P_{t,k} = \frac{\mu_k}{\lambda_{k,\mathcal{P}}(t)}


    Parameters
    ----------
    t : float | array-like
        The time(s) to compute p_tk at.

    intensity : instance of model.Intensity
        Intensity function.

    driver_delays : list of scipy.sparse.csr_matrix
        A list of csr matrices, each one containing, for each driver,
        the delays between the time t and the driver timestamps. Defaults to
        None.

    Returns
    -------
    float | numpy.array
    """

    return intensity.baseline / intensity(t, driver_delays=driver_delays)


def compute_p_tp(t, intensity, driver_delays=None):
    r"""Compute the probability that the activation at time $t$ has been
    triggered by each one of the driver.

    .. math::

        \forall p \in \mathcal{P}, P_{t,p} = \
        \frac{\
        \alpha_{k,p}\sum_{i,t_i^{(p)}<t}\kappa_{k,p}\left(t - t_i^{(p)}\right)
        }{\
        \lambda_{k,\mathcal{P}}(t)\
        }

    Parameters
    ----------
    t : float | array-like
        The time(s) to compute p_tp at.

    intensity : instance of model.Intensity
        Intensity function.

    driver_delays : list of scipy.sparse.csr_matrix
        A list of csr matrices, each one containing, for each driver,
        the delays between the time t and the driver timestamps. Defaults to
        None.

    Returns
    -------
    float | numpy.array
    """

    t = np.atleast_1d(t)

    if driver_delays is None:
        driver_delays = get_driver_delays(intensity, t)

    # compute value of intensity function at each time t
    intensity_at_t = intensity(t, driver_delays=driver_delays)
    # for every driver, compute the associated P_tp
    list_p_tp = []
    for p, delays in enumerate(driver_delays):
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
        val = delays.copy()
        # compute the kernel value at all the "good" delays pre-computed
        val.data = kernel(val.data)
        p_tp = alpha * np.array(val.sum(axis=1).T)[0]
        p_tp /= intensity_at_t
        list_p_tp.append(p_tp)

    return np.array(list_p_tp)


def compute_next_baseline(intensity, T):
    r"""Compute the value of mu at the next EM step.

    .. math::

        \mu_k^{(n+1)} = \frac{1}{T} \sum_{t\in\mathcal{A}_k} P_{t,k}^{(n)}

    Parameters
    ----------
    intensity : instance of model.Intensity
        Intensity function.

    T : int | float
        Total duration of the process, in seconds.

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
    """Compute next values of parameters alpha, m and sigma.

    Parameters
    ----------
    intensity : instance of model.Intensity
        Intensity function.

    C, C_m, C_sigma : float
        Values of constants.

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

    # projection on R+ is eventually done after, during EM
    next_alpha = sum_p_tp / n_drivers_tt

    next_m, next_sigma = [], []
    for p, diff in enumerate(intensity.driver_delays):
        if next_alpha[p] == 0:
            next_m.append(intensity.kernel[p].m)
            next_sigma.append(intensity.kernel[p].sigma)
        else:
            # next value of m for p-th driver
            if C[p] > 0:  # avoid division by 0
                # sum over the driver events
                val = diff.copy()
                val.data *= intensity.kernel[p](val.data)
                sum_temp_m = np.array(val.sum(axis=1).T)[0]
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
                sum_temp_sigma = np.array(val.sum(axis=1).T)[0]
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
    """Compute next values for the 4 parameters.

    Parameters
    ----------
    intensity : instance of model.Intensity
        Intensity function.

    T : int | float
        Total duration of the process, in seconds.


    Returns
    -------
    tuple of 4 floats
        Next value for baseline, alpha, m, sigma.
    """

    baseline = compute_next_baseline(intensity, T)
    C, C_m, C_sigma = compute_Cs(intensity.kernel)
    alpha, m, sigma = compute_next_alpha_m_sigma(intensity, C, C_m, C_sigma)

    return baseline, alpha, m, sigma
