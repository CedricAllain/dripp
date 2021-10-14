"""
From a common initialisation, compare results obtained with DriPP univariate
and DriPP multivariate

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.optim import initialize, em_truncated_norm

from dripp_uni.trunc_norm_kernel.simu import simulate_data as simulate_data_uni
from dripp_uni.trunc_norm_kernel.optim import initialize as initialize_uni
from dripp_uni.trunc_norm_kernel.optim import em_truncated_norm as em_truncated_norm_uni

# %% ------ Simulate data (with new method) ------
N_DRIVERS = 1
SEEDS = range(8)
T = 1_000  # process time, in seconds
ISI = 1
N_TASKS = 0.3
SFREQ = 10_000.
N_ITER = 50
VERBOSE = False
lower, upper = 30e-3, 800e-3
true_params = {'baseline': 0.8,
               'alpha': 0.8,
               'm': 300e-3,
               'sigma': 0.2}

fig, axes = plt.subplots(nrows=2, ncols=2)
# plot true values
for ax, true_value in zip(axes.flatten(), true_params.values()):
    ax.axhline(true_value, 0, N_ITER, color='black', linestyle="dashed")

for i, seed in enumerate(SEEDS):
    # %% ------ Simulate data (with new method) ------
    driver_tt, acti_tt, _, _ = simulate_data(
        lower=lower, upper=upper,
        m=true_params['m'], sigma=true_params['sigma'],
        sfreq=SFREQ,
        baseline=true_params['baseline'], alpha=true_params['alpha'],
        T=T, isi=ISI, n_tasks=N_TASKS,
        n_drivers=N_DRIVERS, seed=seed, return_nll=False, verbose=VERBOSE)

    # driver_tt_uni, acti_tt_uni, _ = simulate_data_uni(
    #     lower=lower, upper=upper,
    #     m=true_params['m'], sigma=true_params['sigma'],
    #     sfreq=SFREQ,
    #     baseline=true_params['baseline'], alpha=true_params['alpha'],
    #     T=T, isi=ISI, n_tasks=N_TASKS,
    #     seed=seed, verbose=VERBOSE)

    # %% ------ Initiate parameters (with new method) ------
    init_params = initialize(acti_tt, driver_tt, lower, upper, T)
    # if VERBOSE:
    #     print("Initial parameters: ", init_params)
    # init_params_uni = (init_params[0], init_params[1][0],
    #                    init_params[2][0], init_params[3][0])
    # init_params_uni = initialize_uni(
    #     acti_tt_uni, driver_tt_uni, lower, upper, T)
    # %% ------ Run DriPP univaraite ------

    # res_uni = em_truncated_norm_uni(acti_tt=acti_tt_uni, driver_tt=driver_tt_uni,
    #                                 lower=lower, upper=upper, T=T, sfreq=SFREQ,
    #                                 init_params=init_params_uni,
    #                                 alpha_pos=True,
    #                                 n_iter=N_ITER,
    #                                 verbose=VERBOSE, disable_tqdm=False)
    # res_uni = em_truncated_norm(acti_tt=acti_tt_uni, driver_tt=driver_tt_uni,
    #                             lower=lower, upper=upper, T=T, sfreq=SFREQ,
    #                             init_params=init_params,
    #                             alpha_pos=True,
    #                             n_iter=N_ITER,
    #                             verbose=VERBOSE, disable_tqdm=False)
    res_multi = em_truncated_norm(acti_tt=acti_tt, driver_tt=driver_tt,
                                  lower=lower, upper=upper, T=T, sfreq=SFREQ,
                                  init_params=init_params,
                                  alpha_pos=True,
                                  n_iter=N_ITER,
                                  verbose=VERBOSE, disable_tqdm=False)

    # df_hist_uni = pd.DataFrame(res_uni[1])
    # for col in ['alpha', 'm', 'sigma']:
    #     df_hist_uni[col] = df_hist_uni[col].apply(lambda x: x[0])
    df_hist_multi = pd.DataFrame(res_multi[1])
    for col in ['alpha', 'm', 'sigma']:
        df_hist_multi[col] = df_hist_multi[col].apply(lambda x: x[0])
    if i == 0:
        legend = True
    else:
        legend = False

    # df_hist_uni.plot(subplots=True, ax=axes, legend=legend, alpha=0.3)
    df_hist_multi.plot(subplots=True, ax=axes, legend=legend,
                       alpha=0.3, linestyle="dashed")


plt.tight_layout()
plt.show()
# %%
