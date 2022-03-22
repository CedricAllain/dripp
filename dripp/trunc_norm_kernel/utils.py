"""Utils functions used for driven point process
with truncated normal kernels.
"""

import itertools
import numbers
import numpy as np
from scipy.sparse import csr_matrix


def convert_variable_multi(var, n=1, repeat=True):
    """Take a variable and make it an array of length n.

    Parameters
    ----------
    var : int | float | array-like
        The variable we want to convert.

    n : int
        The length of the return array. Defauls to 1.

    repeat : bool
        If True, if var is of dimension 1 and n > 1, return an array of
        repeated var of length n. Defaults to True.

    Returns
    -------
    1d-array
    """
    var = np.atleast_1d(var)

    if len(var) == n:  # already at the good size
        return var
    elif repeat and (n > 1) and (len(var) == 1):
        var = np.repeat(var, n)

    assert len(var) == n, \
        "var must be an int, float or an array of length %i, but got %s" % (
            n, var)

    return var


def check_truncation_values(lower, upper):
    """Ensure that the truncation values are ordered"""
    if not upper > lower:
        raise ValueError(
            "truncation values must be sorted, with lower < upper")


def check_driver_tt(driver_tt):
    """Ensure that driver_tt is a list of 1d-arrays"""

    if driver_tt is None:
        return [np.array([])]

    if isinstance(driver_tt[0], numbers.Number):
        driver_tt = [driver_tt]

    driver_tt = [np.array(x) for x in driver_tt]

    return driver_tt


def check_acti_tt(acti_tt):
    """Ensure that acti_tt is a 1d-array"""
    if acti_tt is None:
        return np.array([])

    return np.atleast_1d(acti_tt)


def get_last_timestamps(timestamps, t):
    """Given a time t, find its corresponding last timestamps in a given a list
    of timestamps.

    Parameters
    ----------
    timestamps : array-like
        the array to look in

    t : int | float | array-like
        the value(s) we would like to find the corresponding last timestamps

    Returns
    -------
    last_tmstp : float | numpy.array of floats

    """

    if timestamps == []:
        return np.array([[]])

    if isinstance(timestamps[0], numbers.Number):
        timestamps = np.atleast_2d(timestamps)

    last_tmstp = []

    for tt in timestamps:
        if tt == []:
            this_last_tmstp = np.full(len(t), fill_value=np.nan)
            last_tmstp.append(this_last_tmstp)
            continue
        tt = np.sort(np.array(tt).astype(float))
        # i such that tt[i-1] <= t < tt[i]
        idx = np.searchsorted(tt, t, side='right')
        this_last_tmstp = np.array(tt[idx - 1])
        this_last_tmstp[idx == 0] = np.nan
        last_tmstp.append(this_last_tmstp)

    return np.array(last_tmstp)


def get_driver_delays(intensity, t):
    """For each driver, compute the sparse delay matrix between the time(s) t
    and the driver timestamps.

    Parameters
    ----------
    intensity : instance of model.Intensity

    t : int | float | array-like

    Returns
    -------
    list of scipy.sparse.csr_matrix
    """

    t = np.atleast_1d(t)

    delays = []
    for p in range(intensity.n_drivers):
        driver_tt = intensity.driver_tt[p]
        lower, upper = intensity.kernel[p].lower, intensity.kernel[p].upper
        lower = max(0, lower)
        upper = max(0, upper)
        # Construct a sparse matrix
        this_driver_delays = []
        indices = []
        indptr = [0]
        n_col = 1  # number of columns of the full sparse matrix
        for this_t in t:
            this_t_delays = this_t - driver_tt[(
                driver_tt >= this_t - upper) & ((driver_tt <= this_t - lower))]
            n_delays = len(this_t_delays)
            n_col = max(n_col, n_delays)
            if n_delays > 0:
                this_driver_delays.extend(this_t_delays)
                indices.extend(list(range(n_delays)))

            indptr.append(indptr[-1] + n_delays)

        n_delays = len(this_driver_delays)
        if indptr[-1] != n_delays:
            indptr.append(n_delays)  # add termination

        # create sparse matrix
        M = csr_matrix((np.array(this_driver_delays), np.array(
            indices), np.array(indptr)), shape=(len(t), n_col))
        delays.append(M)

    return delays


if __name__ == '__main__':

    from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity

    # define 2 kernel functions
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = [TruncNormKernel(lower, upper, m, sigma),
              TruncNormKernel(lower, upper, m, sigma)]
    driver_tt = [[3.4, 5.1, 8, 10],
                 [0.5, 2, 4]]  # make sure it respects non-overlapping
    # define intensity function
    baseline, alpha = 0.8, [1.2, 0.5]
    intensity = Intensity(baseline, alpha, kernel, driver_tt)
