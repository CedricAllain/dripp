""" Utils functions for metrics """

import numpy as np
from scipy import integrate


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

    driver_last_tt = intensity.driver_tt[-1]
    assert T >= driver_last_tt, "T is smaller than the last driver's timestamp"

    nll = T * intensity.baseline

    if intensity.alpha == 0:
        nll -= intensity.driver_tt.size * np.log(intensity.baseline)
        return nll

    # check edge effects
    if T >= driver_last_tt + intensity.kernel.upper:
        nll += intensity.alpha * intensity.driver_tt.size
    elif T <= driver_last_tt + intensity.kernel.lower:
        nll += intensity.alpha * (intensity.driver_tt.size - 1)
    else:  # T lands in kernel support of driver's last timestamp
        nll += intensity.alpha * (intensity.driver_tt.size - 1)
        nll += integrate.quad(intensity.kernel,
                              intensity.kernel.lower,
                              T - driver_last_tt)[0]

    nll -= np.log(intensity(intensity.acti_tt,
                            last_tt=intensity.acti_last_tt)).sum()

    return nll
