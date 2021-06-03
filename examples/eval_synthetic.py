"""
Plot figures of results on synthetic data
(figures 2, 3, A.1 and A.2 in the paper)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dripp.experiments.run_multiple_em_on_synthetic import \
    run_multiple_em_on_synthetic
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel

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

N_JOBS = 40  # number of jobs to run in parallel. To adjust based on machine

# ===================================================
# DARA SIMULATION PARAMETERS
# ===================================================

# default parameters for data simulation
simu_params = {'lower': 30e-3, 'upper': 800e-3,
               'baseline': 0.8, 'alpha': 0.8, 'm': 200e-3,
               'isi': 1}

# parameters to vary for data simulation
simu_params_to_vary = {
    'sigma': [0.05, 0.2],  # "sharp" and "wide" kernel scenarios
    'seed': list(range(50)),
    'n_tasks': [0.2, 0.5, 0.8]
}

# ===================================================
# EM PARAMETERS
# ===================================================

# default parameters for EM computation
em_params = {'lower': simu_params['lower'], 'upper': simu_params['upper'],
             'initializer': 'smart_start', 'n_iter': 200, 'alpha_pos': True,
             'verbose': False}

# parameters to vary for EM computation
em_params_to_vary = {'T': np.logspace(2, 5, num=10).astype(int)}

# ===================================================
# RUN EM ON MULTIPLE COMBINATIONS
# ===================================================

# run RM
df_res = run_multiple_em_on_synthetic(
    simu_params, simu_params_to_vary, em_params, em_params_to_vary,
    sfreq=1000, n_jobs=N_JOBS)

# ===================================================
# PLOT FIGURES
# ===================================================


comb1 = {'sigma': 0.2, **simu_params}  # "wide" kernel scenario
comb2 = {'sigma': 0.05, **simu_params}  # "sharp" kernel scenario
combs = [comb1, comb2]

# ------ PLOT KERNEL RETRIEVAL ------
# (Figure 2 in paper)

T = 1000
n_tasks = 0.5
rng = np.random.RandomState(0)
list_seeds = rng.choice(list(range(50)), size=8, replace=False)

fig, axes = plt.subplots(1, 2, figsize=figsize)

start, stop = 0, 1
xx = np.linspace(start, stop, (stop - start) * 300)

for i, this_comb in enumerate(combs):
    # filter resulsts dataframe
    df_comb = df_res[(df_res['T'] == T) &
                     (df_res['n_tasks'] == n_tasks) &
                     (df_res['lower'] == this_comb['lower']) &
                     (df_res['upper'] == this_comb['upper']) &
                     (df_res['baseline'] == this_comb['baseline']) &
                     (df_res['alpha'] == this_comb['alpha']) &
                     (df_res['m'] == this_comb['m']) &
                     (df_res['sigma'] == this_comb['sigma'])]

    # define true kernel
    kernel_true = TruncNormKernel(
        this_comb['lower'], this_comb['upper'],
        this_comb['m'], this_comb['sigma'])

    yy = this_comb['baseline'] + kernel_true.eval(xx)
    axes[i].plot(xx, yy, label='Ground truth', color='black', linestyle='--')

    # define estimated kernels
    for j, seed in enumerate(list_seeds):
        sub_df = df_comb[df_comb['seed'] == seed]
        kernel = TruncNormKernel(this_comb['lower'], this_comb['upper'],
                                 sub_df['m_hat'].values[0],
                                 sub_df['sigma_hat'].values[0])

        yy = sub_df['baseline_hat'].values[0] + kernel.eval(xx)

        if j == 0:
            axes[i].plot(xx, yy, color='blue', alpha=0.2,
                         label='Estimated')
        else:
            axes[i].plot(xx, yy, color='blue', alpha=0.2)

    axes[i].set_xlim(0, stop)
    axes[i].set_xlabel('Time (s)', fontsize=fontsize)
    title = rf"$\mu={this_comb['baseline']}$, "
    title += rf" $\alpha={this_comb['alpha']}$, "
    title += rf" $m={this_comb['m']}$, "
    title += rf" $\sigma={this_comb['sigma']}$"
    axes[i].legend(fontsize=fontsize)
    axes[i].set_title(title, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig2.pdf', dpi=300)
plt.show()
plt.close()

# ------ PLOT MEAN/STD OF THE RELATIVE NORM ------
# (Figure 3 and A.1 in paper)

list_df_mean = []
list_df_std = []
list_title = []

for i, this_comb in enumerate(combs):
    # filter resulsts dataframe
    df_comb = df_res[(df_res['lower'] == this_comb['lower']) &
                     (df_res['upper'] == this_comb['upper']) &
                     (df_res['baseline'] == this_comb['baseline']) &
                     (df_res['alpha'] == this_comb['alpha']) &
                     (df_res['m'] == this_comb['m']) &
                     (df_res['sigma'] == this_comb['sigma'])]

    n_tasks_str = r'$n_t$'
    df_comb = df_comb.rename({'n_tasks': r'$n_t$'}, axis=1)

    # mean
    df_mean = df_comb[['T', n_tasks_str, 'infinite_norm_of_diff_rel']].groupby(
        ['T', n_tasks_str]).mean().unstack()
    df_mean.columns = df_mean.columns.droplevel()
    list_df_mean.append(df_mean)

    # std
    df_std = df_comb[['T', n_tasks_str, 'infinite_norm_of_diff_rel']].groupby(
        ['T', n_tasks_str]).std().unstack()
    df_std.columns = df_std.columns.droplevel()
    list_df_std.append(df_std)

    # title
    title = rf"$\mu={this_comb['baseline']}$, "
    title += rf" $\alpha={this_comb['alpha']}$, "
    title += rf" $m={this_comb['m']}$, "
    title += rf" $\sigma={this_comb['sigma']}$"
    list_title.append(title)


# plot mean (Figure 3 in paper)
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize)

for ii in range(2):
    ax = axes[ii]
    list_df_mean[ii].plot(logy=True, logx=True, cmap=cmap, ax=ax)
    ax.set_xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
    ax.set_ylabel(r"Mean $\|\ \|_{\infty} / \lambda^*_{max}$",
                  fontsize=fontsize)
    ax.set_title(list_title[ii], fontsize=fontsize, pad=5)

for ax in axes.ravel():
    ax.legend([], frameon=False)
    ax.set_xlim(df_comb['T'].min(), df_comb['T'].max())

axes[1].legend(ncol=1, handlelength=1, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig3_mean.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot std (Figure A.1 in paper)
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize)

for ii in range(2):
    ax = axes[ii]
    list_df_std[ii].plot(logy=True, logx=True, cmap=cmap, ax=ax)
    ax.set_xlabel(r"$T$ (s)", fontsize=fontsize, labelpad=0)
    ax.set_ylabel(r"STD $\|\ \|_{\infty} / \lambda^*_{max}$",
                  fontsize=fontsize)
    ax.set_title(list_title[ii], fontsize=fontsize, pad=5)

for ax in axes.ravel():
    ax.legend([], frameon=False)
    ax.set_xlim(df_comb['T'].min(), df_comb['T'].max())

axes[1].legend(ncol=1, handlelength=1, fontsize=fontsize)

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig3_std.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ------ PLOT COMPUTATION TIME OF THE RELATIVE NORM ------
# (Figure A.2 in paper)

plt.figure(figsize=figsize)

ax = sns.lineplot(data=df_res, x="T", y="comput_time",
                  markers=["."], estimator='mean', ci=95)
ax.set_xlim(df_comb['T'].min(), df_comb['T'].max())
ax.set_xlabel(r"$T$ (s)", fontsize=fontsize)
ax.set_ylabel('CPU time (s)', fontsize=fontsize)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(SAVE_RESULTS_PATH / 'fig3_computation_time.pdf', dpi=300)
plt.show()
plt.close()
