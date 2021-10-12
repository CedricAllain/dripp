""" Utils functions used for driven point process
with truncated normal kernels """

import numpy as np


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

    timestamps = np.asarray(timestamps).astype(float)
    timestamps = np.sort(timestamps)

    # i such that timestamps[i-1] <= t < timestamps[i]
    idx = np.searchsorted(timestamps, t, side='right')
    last_tmstp = np.array(timestamps[idx - 1])
    last_tmstp[idx == 0] = np.nan

    return last_tmstp
