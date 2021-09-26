"""
Evaluate DriPP in the multivariate (i.e., multiple drivers) case
"""

# %%

from dripp.trunc_norm_kernel.em import compute_nexts
from dripp.trunc_norm_kernel.optim import initialize
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dripp.experiments.run_multiple_em_on_synthetic import \
    run_multiple_em_on_synthetic
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel
from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.optim import em_truncated_norm

# ===================================================
# PLOT PARAMETERS
# ===================================================

fontsize = 9
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

figsize = (5.5, 2)
cmap = 'viridis_r'

# %%

# %%
# ======================================================
# First, see if ok with only one driver
# ======================================================

# n_drivers = 2

# default parameters for data simulation
# simu_params = {'lower': 30e-3, 'upper': 800e-3,
#                'baseline': 0.8,
#                'alpha': [1.2, 0.2],
#                'm': [200e-3, 50e-3],
#                'sigma': [0.1, 0.5],
#                'sfreq': 150., 'T': 10e5,  # 4 minutes
#                'isi': 1,
#                'n_tasks': 0.2}

# # simulate data with one driver
# driver_tt, acti_tt = simulate_data(
#     n_drivers=2, seed=None, return_nll=False, **simu_params)

# # %%

# em_res = em_truncated_norm(
#     acti_tt, driver_tt, lower=simu_params['lower'], upper=simu_params['upper'],
#     T=simu_params['T'], sfreq=simu_params['sfreq'],
#     initializer='smart_start', n_iter=80, verbose=False, disable_tqdm=False)
# res_params = em_res[0]
# print(res_params)
# # %%

# lower = simu_params['lower']
# upper = simu_params['upper']
# T = simu_params['T']
# sfreq = simu_params['sfreq']
# initializer = 'smart_start'
# n_iter = 80

# acti_tt = np.atleast_1d(acti_tt)
# driver_tt = np.array([np.array(x) for x in driver_tt])
# n_driver = driver_tt.shape[0]
# init_params = initialize(
#     acti_tt, driver_tt, lower, upper, T, initializer=initializer)

# baseline_hat, alpha_hat, m_hat, sigma_hat = init_params
# kernel = []
# for i in range(n_driver):
#     kernel.append(TruncNormKernel(
#         lower, upper, m_hat[i], sigma_hat[i], sfreq=sfreq))
# intensity = Intensity(baseline_hat, alpha_hat, kernel, driver_tt, acti_tt)

# # %%

# nexts = compute_nexts(intensity, T)
# baseline_hat, alpha_hat, m_hat, sigma_hat = nexts
# print(nexts)
# for i in range(n_driver):
#     kernel[i].update(m=m_hat[i], sigma=sigma_hat[i])
# # update intensity function
# intensity.baseline = baseline_hat
# intensity.alpha = alpha_hat
# intensity.kernel = kernel

# %%
# ===================================================
# DATA SIMULATION PARAMETERS
# ===================================================


N_DRIVERS = 2
N_JOBS = 40

# %%

# default parameters for data simulation
simu_params = {'lower': 30e-3, 'upper': 800e-3,
               'baseline': 0.8,
               'alpha': [1.2, 0.4],
               'm': [200e-3, 400e-3],
               'sigma': [0.2, 0.05],
               'isi': 1}

# parameters to vary for data simulation
simu_params_to_vary = {
    # 'seed': list(range(50)),
    'seed': list(range(30)),
    'n_tasks': [0.1, 0.2, 0.4]
}

assert np.max(simu_params_to_vary['n_tasks']) * N_DRIVERS <= 1


# default parameters for EM computation
em_params = {'lower': simu_params['lower'], 'upper': simu_params['upper'],
             'initializer': 'smart_start', 'n_iter': 200, 'alpha_pos': True,
             'verbose': False}

# parameters to vary for EM computation
# em_params_to_vary = {'T': np.logspace(2, 5, num=10).astype(int)}
em_params_to_vary = {'T': np.logspace(2, 4, num=10).astype(int)}
T_max = em_params_to_vary['T'].max()

df_res = run_multiple_em_on_synthetic(
    simu_params, simu_params_to_vary, em_params, em_params_to_vary,
    sfreq=150, n_jobs=N_JOBS, n_drivers=N_DRIVERS, save_results=True)

# %%
# ------ PLOT MEAN/STD OF THE RELATIVE NORM ------
# (Figure 3 and A.1 in paper)

list_df_mean = []
list_df_std = []
list_title = []

n_tasks_str = r'$n_t$'
# n_tasks_str = 'n_t'
df_temp = df_res.rename({'n_tasks': n_tasks_str}, axis=1)

# mean
for p in range(N_DRIVERS):
    columns = ['T', n_tasks_str, 'infinite_norm_of_diff_rel_kernel_%i' % p]
    df_mean = df_temp[columns].groupby(
        ['T', n_tasks_str]).mean().unstack()
    df_mean.columns = df_mean.columns.droplevel()
    # df_mean.plot(logy=True, logx=True, cmap=cmap)
    list_df_mean.append(df_mean)

    df_std = df_temp[columns].groupby(
        ['T', n_tasks_str]).std().unstack()
    df_std.columns = df_std.columns.droplevel()
    list_df_std.append(df_std)

    # title
    title = rf"$\mu={simu_params['baseline']}$, "
    title += rf" $\alpha={simu_params['alpha'][p]}$, "
    title += rf" $m={simu_params['m'][p]}$, "
    title += rf" $\sigma={simu_params['sigma'][p]}$"
    list_title.append(title)


fig, axes = plt.subplots(1, N_DRIVERS, sharey=True,
                         sharex=True, figsize=figsize)

for ii in range(N_DRIVERS):
    ax = axes[ii]
    list_df_mean[ii].plot(logy=True, logx=True, cmap=cmap, ax=ax)
    ax.set_xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
    ax.set_ylabel(r"Mean $\|\ \|_{\infty} / \lambda^*_{max}$",
                  fontsize=fontsize)
    ax.set_title(list_title[ii], fontsize=fontsize, pad=5)

for ax in axes.ravel():
    ax.legend([], frameon=False)
    ax.set_xlim(em_params_to_vary['T'].min(), em_params_to_vary['T'].max())

axes[1].legend(ncol=1, handlelength=1, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig3_mean_multi_%i.pdf' % T_max,
            dpi=300, bbox_inches='tight')
plt.savefig(SAVE_RESULTS_PATH / 'fig3_mean_multi_%i.png' % T_max,
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot std
fig, axes = plt.subplots(1, N_DRIVERS, sharey=True,
                         sharex=True, figsize=figsize)

for ii in range(N_DRIVERS):
    ax = axes[ii]
    list_df_std[ii].plot(logy=True, logx=True, cmap=cmap, ax=ax)
    ax.set_xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
    ax.set_ylabel(r"STD $\|\ \|_{\infty} / \lambda^*_{max}$",
                  fontsize=fontsize)
    ax.set_title(list_title[ii], fontsize=fontsize, pad=5)

for ax in axes.ravel():
    ax.legend([], frameon=False)
    ax.set_xlim(em_params_to_vary['T'].min(), em_params_to_vary['T'].max())

axes[1].legend(ncol=1, handlelength=1, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig3_std_multi_%i.pdf' % T_max,
            dpi=300, bbox_inches='tight')
plt.savefig(SAVE_RESULTS_PATH / 'fig3_std_multi_%i.png' % T_max,
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# plt.xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
# plt.ylabel(r"Mean $\|\ \|_{\infty} / \lambda^*_{max}$",
#            fontsize=fontsize)
# plt.tight_layout()
# plt.savefig(SAVE_RESULTS_PATH / 'fig3_mean_multi.pdf',
#             dpi=300, bbox_inches='tight')
# plt.savefig(SAVE_RESULTS_PATH / 'fig3_mean_multi.png',
#             dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# # std
# for p in range(N_DRIVERS):
#     columns = ['T', n_tasks_str, 'infinite_norm_of_diff_rel_kernel_%i' % p]
#     df_std = df_temp[columns].groupby(
#         ['T', n_tasks_str]).std().unstack()
#     df_std.columns = df_std.columns.droplevel()
#     df_std.plot(logy=True, logx=True, cmap=cmap)

# plt.xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
# plt.ylabel(r"STD $\|\ \|_{\infty} / \lambda^*_{max}$",
#            fontsize=fontsize)
# plt.tight_layout()
# plt.savefig(SAVE_RESULTS_PATH / 'fig3_std_multi.pdf',
#             dpi=300, bbox_inches='tight')
# plt.savefig(SAVE_RESULTS_PATH / 'fig3_std_multi.png',
#             dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()
