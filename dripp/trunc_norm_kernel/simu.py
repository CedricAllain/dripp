"""Utils functions used for the simulation of driven point process
with truncated Gaussian kernel.
"""
# %%
import numpy as np
import time

from dripp.trunc_norm_kernel.utils import convert_variable_multi
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood

from dripp.utils import profile_this


# =======================================
# DRIVER EVENTS TIMESTAMPS
# =======================================

def simu_timestamps_reg(start=0, T=240, isi=1, n_tasks=None, seed=None,
                        add_jitter=False, verbose=False):
    """Simulate regular timestamps with a given isi.

    Simulate regular timestamps over the interval
    [start + 2 * isi ; start + T - 2 * isi],
    with a inter stimuli interval of `isi`,
    and keep `n_tasks` (or `n_tasks` %) of those simulated timestamps,
    sampled uniformly.

    Parameters
    ----------
    start : int
        Starting time. Defaults to 0.

    T : int | float
        Total duration, in seconds. Defaults to 240.

    isi : int | float
        Inter stimuli interval, in seconds. Defaults to 1.

    n_tasks: float | int | None
        Number or percentage of timestamps to keep. If None, keep them all
        (proportion of 1). If float between 0 and 1, act as a percentage. If
        float > 1, then it is converted in int. Defaults to None.

    seed : int
        The numpy seed used for ramdom sampling. Defaults to None.

    add_jitter : bool
        If True, add some random jitter to the selected timestamps. Defaults
        to False.

    verbose : bool
        If True, print some information linked to the timestamps generation. 
        Defaults to False.

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
        timestamps = rng.choice(timestamps, size=n_tasks,
                                replace=False).astype(float)
        if add_jitter:
            jitters = rng.uniform(low=-isi*0.4, high=isi*0.4, size=n_tasks)
            timestamps += jitters
        timestamps.sort()

    if verbose:
        print("%i timestamps generated" % len(timestamps))

    return timestamps


# =======================================
# PROCESS EVENTS TIMESTAMPS
# =======================================

def simu_1d_nonhomogeneous_poisson_process(intensity, T=240, seed=None,
                                           verbose=False):
    """Simulate an Inhomogeneous Poisson Process.

    Simulate an Inhomogeneous Poisson Process with Bounded Intensity Function
    λ(t), on [0, T].
    Source: PA W Lewis and Gerald S Shedler, 1979, "Simulation of
    nonhomogeneous poisson processes by thinning."

    Parameters
    ----------
    intensity: intance of model.Intensity

    T: int | float
        Duration of the Poisson process, in seconds. Defaults to 240.

    seed : int
        The numpy seed used for ramdom sampling. Defaults to None.

    verbose : bool
        If True, print some information. Defaults to False.

    Returns
    -------
    1d numpy.array
        the timestamps of the point process' events
    """

    assert T > 0, "The time duration must be stricly positive"

    lambda_max = intensity.get_max()

    rng = np.random.RandomState(seed)

    tt_list = []
    s = 0.
    while s <= T:
        u = rng.rand()
        w = -np.log(u) / lambda_max  # w drawn from Exp(lambda_max)
        s += w
        D = rng.rand()
        if D <= intensity(s) / lambda_max:  # if intensity(s) < 0, no event
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

@profile_this
def simulate_data(lower=30e-3, upper=500e-3, m=150e-3, sigma=0.1, sfreq=150.,
                  baseline=0.8, alpha=1, T=240, isi=1, add_jitter=False,
                  n_tasks=0.8, n_drivers=1, seed=None, return_nll=True,
                  verbose=False):
    """Given some initial parameters, simulate driver's timestamps and
    driven process's activation timestamps, for the intensity defined with a
    truncated gaussian kernel.

    Parameters
    ----------
    lower, upper : int | float | array-like
        Kernel's truncation values. If lower and upper are int or float with
        n_drivers > 1, their values are shared accross all generated driver; if
        they are array-like, they must be of length n_drivers. Defaults to
        lower = 30e-3 and upper = 500e-3.

    m, sigma : int | float | array-like
        Kernel's shape parameters: mean and stadard deviation, respectively
        similarly to lower, upper, either the values are shared accross all
        kernels, or they must be arrays of length n_drivers. Defaults to 
        m = 150e-3 and sigma = 0.1.

    sfreq : int | None
        Sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values. If None, the kernel will be
        exactly evaluate at each call. Warning: setting sfreq to None may
        considerably increase computational time. Defaults to 150.

    baseline : int | float
        Baseline intensity value. Defaults to 0.8.

    alpha : int | float | array-like
        Coefficient of influence of the driver on the process
        similarly to lower, upper, either the value is shared accross all
        kernels, or it must be an array of length n_drivers. Defaults to 1.

    T : int | float
        Duration of the process, in seconds (same duration for all drivers).
        Defaults to 240.

    isi : int | float | array-like
        Inter stimuli interval. Similarly to lower, upper, either the value is
        shared accross all kernels, or it must be an array of length n_drivers.
        Defaults to 1.

    n_tasks: float | int | None
        Number or percentage of timestamps to keep. If None, keep them all
        (proportion of 1). If float between 0 and 1, act as a percentage. If
        float > 1, then it is converted in int. similarly to lower, upper,
        either the value is shared accross all kernels, or it must be an array
        of length n_drivers. Defaults to None.

    n_drivers : int
        Number of driver to simulate. Defaults to 1.

    seed : int | array-like
        The seed used for ramdom sampling. If n_drivers > 1, seed must be None
        or array-like of length n_drivers. Defaults to None.

    return_nll : bool
        If True, compute and return the true negative log-likelihood (nll).
        Defaults to True.

    verbose : bool
        If True, print some information linked to the timestamps generation.
        Defaults to False.


    Returns
    -------
    driver_tt : 1d | 2d numpy.array
        Driver's timestamps. If n_drivers > 1, the second dimension is equal
        to the number of drivers.

    acti_tt : 1d | 2d numpy.array
        Process's activation timestamps. If n_drivers > 1, the second dimension
        is equal to the number of drivers.

    if return_nll is True:
        true_nll : float
            negative log-likelihood of the process

    kernel : list of TruncNormKernel instances

    intensity : intance of model.Intensity
    """
    isi = convert_variable_multi(isi, n_drivers, repeat=True)
    n_tasks = convert_variable_multi(n_tasks, n_drivers, repeat=True)
    seed = convert_variable_multi(seed, n_drivers, repeat=True)
    if seed[0] is not None:
        rng = np.random.RandomState(seed[0])
        seed += rng.choice(range(100), size=n_drivers, replace=False)

    # simulate task timestamps
    driver_tt = []
    for this_isi, this_n_tasks, this_seed in zip(isi, n_tasks, seed):
        this_driver_tt = simu_timestamps_reg(
            T=T, isi=this_isi, n_tasks=this_n_tasks, seed=this_seed,
            add_jitter=add_jitter, verbose=verbose)
        driver_tt.append(this_driver_tt)

    # define kernel and intensity functions
    lower = convert_variable_multi(lower, n_drivers, repeat=True)
    upper = convert_variable_multi(upper, n_drivers, repeat=True)
    m = convert_variable_multi(m, n_drivers, repeat=True)
    sigma = convert_variable_multi(sigma, n_drivers, repeat=True)

    kernel = []
    for this_lower, this_upper, this_m, this_sigma in \
            zip(lower, upper, m, sigma):
        kernel.append(TruncNormKernel(this_lower, this_upper, this_m,
                                      this_sigma, sfreq=sfreq))

    alpha = convert_variable_multi(alpha, n_drivers, repeat=True)
    intensity = Intensity(baseline, alpha, kernel, driver_tt)

    # simulate activation timestamps
    acti_tt = simu_1d_nonhomogeneous_poisson_process(
        intensity=intensity, T=T, seed=seed[0], verbose=verbose)
    intensity.acti_tt = acti_tt

    if return_nll:
        # update intensity function and compute true negative log-likelihood
        intensity.acti_tt = acti_tt
        true_nll = negative_log_likelihood(intensity=intensity, T=T)
        return driver_tt, acti_tt, true_nll, kernel, intensity
    else:
        return driver_tt, acti_tt, kernel, intensity


if __name__ == '__main__':
    N_DRIVERS = 2
    T = 10_000  # process time, in seconds
    lower, upper = 30e-3, 800e-3
    start_time = time.time()
    driver_tt, acti_tt, kernel, intensity = simulate_data(
        lower=lower, upper=upper,
        m=[400e-3, 400e-3], sigma=[0.2, 0.05],
        sfreq=300.,
        baseline=0.5, alpha=[-0.8, 0.8],
        T=T, isi=[1, 1.2], n_tasks=0.8,
        n_drivers=N_DRIVERS, seed=0, return_nll=False, verbose=False)
    simu_time = time.time() - start_time
    print("Simulation time for %i driver(s) over %i seconds: %.3f seconds"
          % (N_DRIVERS, T, simu_time))

    from dripp.trunc_norm_kernel.optim import initialize
    baseline_init, alpha_init, m_init, sigma_init = initialize(intensity, T)

# %%
