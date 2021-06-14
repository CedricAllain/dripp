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

    if timestamps == []:
        return np.array([[]])

    if isinstance(timestamps[0], (int, float)):
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
