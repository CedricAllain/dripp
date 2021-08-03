"""
Utils functions used for the simulation of driven point process
with truncated Gaussian kernel
"""

# %%

import numpy as np

from dripp.trunc_norm_kernel.utils import convert_variable_multi
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood


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
                  n_drivers=1, seed=None, return_nll=True, verbose=False):
    """ Given some initial parameters, simulate driver's timestamps and
    driven process's activation timestamps, for the intensity defined with a
    truncated gaussian kernel.

    Parameters
    ----------
    lower, upper : int | float | array-like
        kernel's truncation values
        if lower and upper are int or float with n_drivers > 1, their values 
        are shared accross all generated driver; if they are array-like, they
        must be of length n_drivers
        default is 30e-3, 500e-3

    m, sigma : int | float | array-like
        kernel's shape parameters: mean and stadard deviation, respectively
        similarly to lower, upper, either the values are shared accross all
        kernels, or they must be arrays of length n_drivers
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

    alpha : int | float | array-like
        coefficient of influence of the driver on the process
        similarly to lower, upper, either the value is shared accross all
        kernels, or it must be an array of length n_drivers
        default is 1

    T : int | float
        duration of the process, in seconds (same duration for all drivers)
        default is 240

    isi : int | float | array-like
        inter stimuli interval
        similarly to lower, upper, either the value is shared accross all
        kernels, or it must be an array of length n_drivers
        default is 1

    n_tasks : float | int | array-like | None
        number or percentage of timestamps to keep
        if None, keep them all
        if float between 0 and 1, act as a percentage
        if float > 1, then it is converted in int
        similarly to lower, upper, either the value is shared accross all
        kernels, or it must be an array of length n_drivers
        default is None

    n_drivers : int
        number of driver to simulate
        default is 1

    seed : int | array-like
        the seed used for ramdom sampling
        if n_drivers > 1, seed must be None or array-like of length n_drivers
        default is None

    return_nll : bool
        if True, compute the true negative log-likelihood

    verbose : bool
        if True, print some information linked to the timestamps generation
        default is False


    Returns
    -------
    driver_tt : 1d | 2d numpy.array
        driver's timestamps
        if n_drivers > 1, the second dimension if n_drivers

    acti_tt : 1d | 2d numpy.array
        process's activation timestamps
        if n_drivers > 1, the second dimension if n_drivers

    true_nll : float
        negative log-likelihood of the process

    """

    isi = convert_variable_multi(isi, n_drivers, repeat=True)
    n_tasks = convert_variable_multi(n_tasks, n_drivers, repeat=True)
    seed = convert_variable_multi(seed, n_drivers, repeat=True)

    # simulate task timestamps
    driver_tt = []
    for this_isi, this_n_tasks, this_seed in zip(isi, n_tasks, seed):
        this_driver_tt = simu_timestamps_reg(
            T=T, isi=this_isi, n_tasks=this_n_tasks, seed=this_seed,
            verbose=verbose)
        driver_tt.append(this_driver_tt)
    driver_tt = np.array([np.array(x) for x in driver_tt], dtype=object)

    # define kernel and intensity functions
    lower = convert_variable_multi(lower, n_drivers, repeat=True)
    upper = convert_variable_multi(upper, n_drivers, repeat=True)
    m = convert_variable_multi(m, n_drivers, repeat=True)
    sigma = convert_variable_multi(sigma, n_drivers, repeat=True)

    kernel = []
    for this_lower, this_upper, this_m, this_sigma in zip(lower, upper, m, sigma):
        kernel.append(TruncNormKernel(this_lower, this_upper, this_m,
                                      this_sigma, sfreq=sfreq))
    intensity = Intensity(baseline, alpha, kernel, driver_tt)

    # simulate activation timestamps
    acti_tt = simu_1d_nonhomogeneous_poisson_process(
        intensity=intensity, T=T, seed=seed[0], verbose=verbose)

    if return_nll:
        # update intensity function and compute true negative log-likelihood
        intensity.acti_tt = acti_tt
        true_nll = negative_log_likelihood(intensity=intensity, T=T)
        return driver_tt, acti_tt, true_nll
    else:
        return driver_tt, acti_tt


if __name__ == '__main__':
    driver_tt, acti_tt, true_nll = simulate_data(
        lower=30e-3, upper=500e-3, m=150e-3, sigma=0.1, sfreq=150.,
        baseline=0.8, alpha=1, T=240, isi=1, n_tasks=0.8,
        n_drivers=2, seed=None, return_nll=True, verbose=False)
    print("true_nll: %.3f" % true_nll)
