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


SAVE_RESULTS_PATH /= 'results_sythetic'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

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
# ===================================================
# DATA SIMULATION PARAMETERS
# ===================================================


N_DRIVERS = 2
N_JOBS = 50
SFREQ = 1000

# default parameters for data simulation
simu_params = {'lower': 30e-3, 'upper': 800e-3,
               'baseline': 0.5,
               'alpha': [0.8, 0.8],
               #    'm': [200e-3, 400e-3],
               'm': [300e-3, 300e-3],
               'sigma': [0.2, 0.05],
               'isi': 1}

# parameters to vary for data simulation
simu_params_to_vary = {
    # 'seed': list(range(30)),
    # 'n_tasks': [0.2, 0.4, 0.8]
    'seed': list(range(8)),
    'n_tasks': [0.4]
}

# assert np.max(simu_params_to_vary['n_tasks']) * N_DRIVERS <= 1


# default parameters for EM computation
em_params = {'lower': simu_params['lower'], 'upper': simu_params['upper'],
             'initializer': 'smart_start', 'n_iter': 200, 'alpha_pos': True,
             'verbose': False}

# parameters to vary for EM computation
# em_params_to_vary = {'T': np.logspace(2, 4, num=7).astype(int)}
em_params_to_vary = {'T': np.array([3000])}
T_max = em_params_to_vary['T'].max()

df_res = run_multiple_em_on_synthetic(
    simu_params, simu_params_to_vary, em_params, em_params_to_vary,
    sfreq=SFREQ, n_jobs=N_JOBS, n_drivers=N_DRIVERS, save_results=True)

# %% ------ PLOT KERNEL RETRIEVAL ------
# (Figure 2 in paper)

T = 3000
n_tasks = 0.4

fig, axes = plt.subplots(1, 2, figsize=figsize)

start, stop = 0, 1
xx = np.linspace(start, stop, (stop - start) * 300)

for i in range(N_DRIVERS):
    # filter resulsts dataframe
    df_comb = df_res[(df_res['T'] == T) &
                     (df_res['n_tasks'] == n_tasks)]

    # define true kernel
    kernel_true = TruncNormKernel(
        simu_params['lower'], simu_params['upper'],
        simu_params['m'][i], simu_params['sigma'][i])

    yy = simu_params['baseline'] + kernel_true.eval(xx)
    axes[i].plot(xx, yy, label='Ground truth', color='black', linestyle='--')

    # plot a few estimated kernels
    for seed in range(8):
        sub_df = df_res[(df_res['T'] == T) &
                        (df_res['n_tasks'] == n_tasks) &
                        (df_res['seed'] == seed)]
        kernel = TruncNormKernel(simu_params['lower'], simu_params['upper'],
                                 list(sub_df['m_hat'])[0][i],
                                 list(sub_df['sigma_hat'])[0][i])

        yy = sub_df['baseline_hat'].values[0] + kernel.eval(xx)

        if seed == 0:
            axes[i].plot(xx, yy, color='blue', alpha=0.2,
                         label='Estimated')
        else:
            axes[i].plot(xx, yy, color='blue', alpha=0.2)

    axes[i].set_xlim(0, stop)
    axes[i].set_xlabel('Time (s)', fontsize=fontsize)
    title = rf"$\mu={simu_params['baseline']}$, "
    title += rf" $\alpha={simu_params['alpha'][i]}$, "
    title += rf" $m={simu_params['m'][i]}$, "
    title += rf" $\sigma={simu_params['sigma'][i]}$"
    axes[i].legend(fontsize=fontsize)
    axes[i].set_title(title, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / ('fig2_multi_%i.pdf' % T), dpi=300)
plt.savefig(SAVE_RESULTS_PATH / ('fig2_multi_%i.png' % T), dpi=300)
plt.show()
plt.close()
# %%
# ------ PLOT MEAN/STD OF THE RELATIVE NORM ------
# (Figure 3 and A.1 in paper)

list_df_mean = []
list_df_std = []
list_title = []

n_tasks_str = r'$n_t$'
# n_tasks_str = 'n_t'
df_temp = df_res.rename({'n_tasks': n_tasks_str}, axis=1)

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


# plot mean
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
fig_name = 'fig3_mean_multi_%i_sfreq_%i' % (T_max, SFREQ)
plt.savefig(SAVE_RESULTS_PATH / (fig_name + '.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(SAVE_RESULTS_PATH / (fig_name + '.png'),
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
fig_name = fig_name.replace('mean', 'std')
plt.savefig(SAVE_RESULTS_PATH / (fig_name + '.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(SAVE_RESULTS_PATH / (fig_name + '.png'),
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
