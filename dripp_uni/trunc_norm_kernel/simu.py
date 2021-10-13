"""
Utils functions used for the simulation of driven point process
with truncated Gaussian kernel
"""

import numpy as np

from .model import TruncNormKernel, Intensity
from .metric import negative_log_likelihood


# =======================================
# DRIVER EVENTS TIMESTAMPS
# =======================================

def simu_timestamps_reg(start=0, T=240, isi=1, n_tasks=None, seed=None,
                        verbose=False):
    """ Simulate regular timestamps over the interval
    [start + 2 * isi ; start + T - 2 * isi],
    with a inter stimuli interval of `isi`,
    and keep `n_tasks` (or `n_tasks` %) of those simulated timestamps,
    sampled uniformly.

    Parameters
    ----------
    start : int
        starting time
        default is 0

    T : int | float
        total duration, in seconds
        default is 240

    isi : int | float
        inter stimuli interval, in seconds
        default is 1

    n_tasks: float | int | None
        number or percentage of timestamps to keep
        if None, keep them all (proportion of 1)
        if float between 0 and 1, act as a percentage
        if float > 1, then it is converted in int
        default is None

    seed : int
        the seed used for ramdom sampling
        default is None

    verbose : bool
        if True, print some information linked to the timestamps generation
        default is False

    Returns
    -------
    1d numpy.array : array of floats, sorted in ascending order
    """

    assert T > 0, "The time duration must be stricly positive"

    assert isi > 0, "The time interval for sampling must be stricly "\
        "positive, but gave isi <= 0."

    timestamps = np.arange(start=start + 2*isi,
                           stop=start + T - 2*isi,
                           step=isi)

    if n_tasks is not None:
        assert n_tasks <= len(timestamps), \
            'n_tasks too large, modify T or isi in order to get n_tasks.'
        assert n_tasks >= 0, \
            "n_tasks must be positive, between 0 and 1 for a percentage,"\
            " or bigger than 1 for an absolute value."

        if n_tasks < 1:
            # act as a percentage of the number of simulated timestamps
            if verbose:
                print("n_tasks interpreted as a percentage: %i%%"
                      % (n_tasks*100))

            n_tasks = int(n_tasks * len(timestamps))

        if type(n_tasks) == float:
            n_tasks = int(n_tasks)
            if verbose:
                print("n_tasks rounded to %i" % n_tasks)

        # sample timestamps
        rng = np.random.RandomState(seed)
        timestamps = rng.choice(timestamps, size=n_tasks, replace=False)
        timestamps.sort()

    if verbose:
        print("%i timestamps generated" % len(timestamps))

    return timestamps


# =======================================
# PROCESS EVENTS TIMESTAMPS
# =======================================

def simu_1d_nonhomogeneous_poisson_process(intensity,
                                           T=240,
                                           lambda_max=None,
                                           seed=None,
                                           verbose=False):
    """Simulation of an Inhomogeneous Poisson Process with
    Bounded Intensity Function λ(t), on [0, T].
    Source: PA W Lewis and Gerald S Shedler, 1979,
    "Simulation of nonhomogeneous poisson processes by thinning."

    Parameters
    ----------
    intensity: callable (defaul None)
        intensity function

    T: int | float
        duration of the Poisson process, in seconds
        default is 240

    lambda_max: int | float
        maximum of the intensity function on [0, T]
        if None, it is computed using intensity.get_max()
        default is None

    seed : int
        the seed used for ramdom sampling
        default is None

    verbose : bool
        if True, print some information
        default is False

    Returns
    -------
    1d numpy.array
        the timestamps of the point process' events

    """

    assert T > 0, "The time duration must be stricly positive"

    if lambda_max is None:
        lambda_max = intensity.get_max()

    rng = np.random.RandomState(seed)

    tt_list = []
    s = 0.
    while s <= T:
        u = rng.rand()
        w = -np.log(u) / lambda_max
        s += w
        D = rng.rand()
        if D <= intensity(s) / lambda_max:
            tt_list.append(s)

    if (tt_list == []) or (tt_list[-1] <= T):
        tt_list = np.array(tt_list)
    else:
        tt_list = np.array(tt_list[:-1])

    if verbose:
        print("%i events generated" % len(tt_list))

    return tt_list


# =======================================
# GENERATE FULL SET OF DATA
# =======================================

def simulate_data(lower=30e-3, upper=500e-3, m=150e-3, sigma=0.1, sfreq=150.,
                  baseline=0.8, alpha=1, T=240, isi=1, n_tasks=0.8,
                  seed=None, verbose=False):
    """ Given some initial parameters, simulate driver's timestamps and
    driven process's activation timestamps, for the intensity defined with a
    truncated gaussian kernel.

    Parameters
    ----------
    lower, upper : int | float
        kernel's truncation values
        default is 30e-3, 500e-3

    m, sigma : int | float
        kernel's shape parameters: mean and stadard deviation, respectively
        default is 150e-3, 0.1

    sfreq : int | None
        sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    baseline : int | float
        baseline intensity
        default is 0.8

    alpha : int | float
        coefficient of influence of the driver on the process
        default is 1

    T : int | float
        duration of the process, in seconds
        default is 240

    isi : int | float
        inter stimuli interval
        default is 1

    n_tasks: float | int | None
        number or percentage of timestamps to keep
        if None, keep them all
        if float between 0 and 1, act as a percentage
        if float > 1, then it is converted in int
        default is None

    seed : int
        the seed used for ramdom sampling
        default is None

    verbose : bool
        if True, print some information linked to the timestamps generation
        default is False


    Returns
    -------
    driver_tt : 1d numpy.array
        driver's timestamps

    acti_tt : 1d numpy.array
        process's activation timestamps

    true_nll : float
        negative log-likelihood of the process

    """

    # simulate task timestamps
    driver_tt = simu_timestamps_reg(
        T=T, isi=isi, n_tasks=n_tasks, seed=seed, verbose=verbose)

    # define kernel and intensity functions
    kernel = TruncNormKernel(lower, upper, m, sigma, sfreq=sfreq)
    intensity = Intensity(baseline, alpha, kernel, driver_tt)

    # simulate activation timestamps
    acti_tt = simu_1d_nonhomogeneous_poisson_process(
        intensity=intensity, T=T, seed=seed, verbose=verbose)

    # update intensity function and compute true negative log-likelihood
    intensity.acti_tt = acti_tt
    true_nll = negative_log_likelihood(intensity=intensity, T=T)

    return driver_tt, acti_tt, true_nll
