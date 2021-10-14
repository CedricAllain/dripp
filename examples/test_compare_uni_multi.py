"""
Compare estimated values between DriPP univariate and DriPP multivariate
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.optim import initialize, em_truncated_norm

from dripp_uni.trunc_norm_kernel.optim import initialize as initialize_uni
from dripp_uni.trunc_norm_kernel.optim import em_truncated_norm as em_truncated_norm_uni

import cProfile


def profile_this(fn):
    def profiled_fn(*args, **kwargs):
        filename = fn.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_fn


@profile_this
def test_compare_uni_multi():

    # ------ Simulate data (with multi method) ------
    N_DRIVERS = 1
    T = 1_000  # process time, in seconds
    ISI = 1
    N_TASKS = 0.8
    SFREQ = 1000.
    N_ITER = 200
    lower, upper = 30e-3, 800e-3
    true_params = {'baseline': 0.8,
                   'alpha': 0.8,
                   'm': 200e-3,
                   'sigma': 0.2}
    driver_tt, acti_tt, kernel, intensity = simulate_data(
        lower=lower, upper=upper,
        m=true_params['m'], sigma=true_params['sigma'],
        sfreq=SFREQ,
        baseline=true_params['baseline'], alpha=true_params['alpha'],
        T=T, isi=ISI, n_tasks=N_TASKS,
        n_drivers=N_DRIVERS, seed=0, return_nll=False, verbose=False)

    # ------ Initiate parameters (with multi method) ------
    init_params = initialize(acti_tt, driver_tt, lower, upper, T)
    init_params_uni = (init_params[0], init_params[1][0],
                       init_params[2][0], init_params[3][0])
    # ------ Run DriPP univaraite ------
    res_uni = em_truncated_norm_uni(acti_tt=acti_tt, driver_tt=driver_tt[0],
                                    lower=lower, upper=upper, T=T, sfreq=SFREQ,
                                    init_params=init_params_uni,
                                    alpha_pos=True,
                                    n_iter=N_ITER,
                                    verbose=False, disable_tqdm=False)
    estimates_uni = np.array(res_uni[0])
    # ------ Run DriPP multivaraite ------
    res_multi = em_truncated_norm(acti_tt=acti_tt, driver_tt=driver_tt,
                                  lower=lower, upper=upper, T=T, sfreq=SFREQ,
                                  init_params=init_params,
                                  alpha_pos=True,
                                  n_iter=N_ITER,
                                  verbose=False, disable_tqdm=False)
    baseline, alpha, m, sigma = res_multi[0]
    estimates_multi = np.array([baseline, alpha[0], m[0], sigma[0]])
    # ------ Compare the two results ------
    np.testing.assert_allclose(estimates_uni, estimates_multi)


if __name__ == '__main__':
    test_compare_uni_multi()
    print('Profile is available in [func.__name__].profile.'
          'Use runsnake or snakeviz to view it')
