"""Utils functions for metrics.
"""

import numpy as np
from scipy import integrate


def negative_log_likelihood_1d(intensity, T):
    """Compute the negative log-likelihood for a given intensity function over
    a given time duration, in the case where the intensity only have one
    driver.

    Parameters
    ----------
    intensity : model.Trun object

    T : float
        Total duration.

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
    """Compute the negative log-likelihood for a given intensity function over
    a given time duration.

    Parameters
    ----------
    intensity : instance of model.Intensity

    T : float
        Total duration.

    Returns
    -------
    float
    """

    nll = T * intensity.baseline

    for this_alpha, this_driver_tt in zip(intensity.alpha, intensity.driver_tt):
        nll += this_alpha * len(this_driver_tt)

    nll -= np.log(intensity(intensity.acti_tt,
                            driver_delays=intensity.driver_delays)).sum()

    return nll


def infinite_norm_intensity(intensity_true, intensity_hat):
    """

    """
    inf_norm_rel = []

    for p in range(intensity_true.n_drivers):
        # get associated kernels
        kernel_true = intensity_true.kernel[p]
        kernel_hat = intensity_hat.kernel[p]
        lower_ = min(kernel_true.lower, kernel_hat.lower)
        upper_ = max(kernel_true.upper, kernel_hat.upper)
        xx = np.linspace(lower_ - 1, upper_ + 1,
                         int(np.ceil(800*(upper_ - lower_ + 2))))
        # true intensity at kernel p
        yy_true = intensity_true.baseline
        if intensity_true.alpha[p] != 0:
            yy_true += intensity_true.alpha[p] * kernel_true.eval(xx)
        # estimated intensity at kernel p
        yy_hat = intensity_hat.baseline
        if intensity_hat.alpha[p] != 0:
            yy_hat += intensity_hat.alpha[p] * kernel_hat.eval(xx)
        # compute infinite norm between true and estimated intensities
        inf_norm = abs(np.atleast_1d(yy_true - yy_hat)).max()
        # compute maximum of true intensity at kernel p
        lambda_max = intensity_true.baseline + \
            intensity_true.alpha[p] * kernel_true.max
        # compute relative infinite norm
        inf_norm_rel.append(inf_norm / lambda_max)

    return inf_norm_rel
