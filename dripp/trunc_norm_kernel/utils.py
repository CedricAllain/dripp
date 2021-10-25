""" Utils functions used for driven point process
with truncated normal kernels """

# %%

from scipy.sparse import csr_matrix
import itertools
import numbers
import numpy as np


def convert_variable_multi(var, n=1, repeat=True):
    """Take a variable and make it an array of length n

    Parameters
    ----------
    var : int | float | array-like
        the variable we want to convert

    n : int
        the length of the return array
        default is 1

    repeat : bool
        if True, if var is of dimension 1 and n > 1, return an array of
        repeated var of length n
        default is True

    Returns
    -------
    var

    """

    var = np.atleast_1d(var)

    if len(var) == n:
        # already at the good size
        return var
    elif repeat and (n > 1) and (len(var) == 1):
        var = np.repeat(var, n)

    assert len(var) == n, \
        "var must be an int, float or an array of length n"

    return var


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

    return np.array(last_tmstp)  # , dtype=object)


def get_driver_tt_of_influence(intensity, t):
    """For a given time and a intensity function, find, for all of its driver, 
    the timestamsp that may still have an influence at that time.
    For a kernels that are truncated gaussians, that means finding all the t_i
    among a driver's timestamps such that t_i + a <= t <= t_i + b

    Parameters
    ----------
    t : int | float 

    intensity : instance of Intensity


    Returns
    -------
    numpy 2d array, filled with np.nan values
    """

    driver_tt_of_influence = []
    for p in range(intensity.n_drivers):
        driver_tt = intensity.driver_tt[p]
        lower, upper = intensity.kernel[p].lower, intensity.kernel[p].upper
        this_driver_tt_of_influence = driver_tt[(
            driver_tt >= t - upper) & ((driver_tt <= t - lower))]
        driver_tt_of_influence.append(this_driver_tt_of_influence)

    driver_tt_of_influence = np.array(
        list(itertools.zip_longest(*driver_tt_of_influence, fillvalue=0))).T

    return driver_tt_of_influence


def get_driver_delays(intensity, t):
    """
    For each driver, compute the delays with t that are on support


    Returns
    -------
    list of 2D numpy.array

    """

    t = np.atleast_1d(t)

    delays = []
    for p in range(intensity.n_drivers):
        driver_tt = intensity.driver_tt[p]
        lower, upper = intensity.kernel[p].lower, intensity.kernel[p].upper
        # Construct a sparse matrix
        this_driver_delays = []
        indices = []
        indptr = [0]
        n_col = 1  # number of colons of the full sparse matrix
        for this_t in t:
            this_t_delays = this_t - driver_tt[(
                driver_tt >= this_t - upper) & ((driver_tt <= this_t - lower))]
            # this_driver_delays.append(this_t_delays)
            n_delays = len(this_t_delays)
            n_col = max(n_col, n_delays)
            if n_delays > 0:
                this_driver_delays.extend(this_t_delays)
                indices.extend(list(range(n_delays)))

            indptr.append(indptr[-1] + n_delays)

        n_delays = len(this_driver_delays)
        if indptr[-1] != n_delays:
            # add termination
            indptr.append(n_delays)

        # create sparse matrix
        M = csr_matrix((np.array(this_driver_delays), np.array(
            indices), np.array(indptr)), shape=(len(t), n_col))
        delays.append(M)

    return delays

# %%


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

    # get delays for t = 0
    t = 0
    # %%
