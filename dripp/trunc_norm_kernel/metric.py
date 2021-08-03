""" Utils functions for metrics """

import numpy as np
from scipy import integrate


def negative_log_likelihood_1d(intensity, T):
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

    nll -= np.log(intensity(acti_tt, last_tt=intensity.acti_last_tt)).sum()

    return nll


def negative_log_likelihood(intensity, T):
    """

    """

    n_drivers = len(intensity.kernel)

    if n_drivers == 1:
        return negative_log_likelihood_1d(intensity, T)

    # in the case of multiple drivers
    nll = T * intensity.baseline
    for this_alpha, this_driver_tt in zip(intensity.alpha, intensity.driver_tt):
        nll += this_alpha * len(this_driver_tt)

    for this_acti_tt in intensity.acti_tt:
        sum_temp = intensity.baseline
        for p in range(n_drivers):
            delays = this_acti_tt - intensity.driver_tt[p]
            # only take into account positive delays
            sum_temp += intensity.alpha[p] * delays[delays > 0].sum()
        nll -= np.log(sum_temp)

    return nll
