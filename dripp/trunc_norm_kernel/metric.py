""" Utils functions for metrics """

import numpy as np
from scipy import integrate


def negative_log_likelihood_1d(intensity, T):
    """ Compute the negative log-likelihood for a given intensity function over
    a given time duration, in the case where the intensity only have one driver

    Parameters
    ----------
    intensity : model.Trun object

    T : float
        total duration

    Returns
    -------
    float
    """

    alpha = intensity.alpha[0]
    kernel = intensity.kernel[0]
    acti_tt = intensity.acti_tt[0]
    driver_tt = intensity.driver_tt[0]

    driver_last_tt = driver_tt[-1]
    assert T >= driver_last_tt, "T is smaller than the last driver's timestamp"

    nll = T * intensity.baseline

    if alpha == 0:
        nll -= driver_tt.size * np.log(intensity.baseline)
        return nll

    # check edge effects
    if T >= driver_last_tt + kernel.upper:
        nll += alpha * driver_tt.size
    elif T <= driver_last_tt + intensity.kernel.lower:
        nll += alpha * (driver_tt.size - 1)
    else:  # T lands in kernel support of driver's last timestamp
        nll += alpha * (driver_tt.size - 1)
        nll += integrate.quad(kernel,
                              kernel.lower,
                              T - driver_last_tt)[0]

    nll -= np.log(intensity(acti_tt)).sum()

    return nll


def negative_log_likelihood(intensity, T):
    """ Compute the negative log-likelihood for a given intensity function over
    a given time duration

    Parameters
    ----------
    intensity : model.Trun object

    T : float
        total duration

    Returns
    -------
    float
    """

    # n_drivers = len(intensity.kernel)

    # if n_drivers == 1:
    #     return negative_log_likelihood_1d(intensity, T)

    # in the case of multiple drivers
    nll = T * intensity.baseline
    for this_alpha, this_driver_tt in zip(intensity.alpha, intensity.driver_tt):
        nll += this_alpha * len(this_driver_tt)

    nll -= np.log(intensity(intensity.acti_tt,
                            driver_delays=intensity.driver_delays)).sum()

    # for this_acti_tt in intensity.acti_tt:
    #     sum_temp = intensity.baseline
    #     for p in range(n_drivers):
    #         # compute delays
    #         delays = this_acti_tt - intensity.driver_tt[p]
    #         # compute the sum of kernel values for the delays
    #         sum_temp += intensity.alpha[p] * intensity.kernel[p](delays).sum()
    #     nll -= np.log(sum_temp)

    return nll
