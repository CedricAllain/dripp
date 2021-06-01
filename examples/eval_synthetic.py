"""
Plot figures of results on synthetic data
(figures 2, 3, A.1 and A.2 in the paper)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dripp.experiments.run_multiple_em_on_simu import run_multiple_em_on_simu
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

# ===================================================
# DARA SIMULATION PARAMETERS
# ===================================================

# default parameters for data simulation
simu_params = {'lower': 30e-3, 'upper': 800e-3,
               'baseline': 0.8, 'alpha': 0.8, 'm': 200e-3,
               'isi': 1, 'uniform': True}

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
             'initializer': 'smart_start', 'n_iter': 200, 'alpha_pos': True}

# parameters to vary for EM computation
em_params_to_vary = {'T': np.logspace(2, 4, num=5).astype(int)}

# ===================================================
# RUN EM ON MULTIPLE COMBINATIONS
# ===================================================

# run RM
df_res = run_multiple_em_on_simu(
    simu_params, simu_params_to_vary, em_params, em_params_to_vary,
    sfreq=1000, n_jobs=50)

# save results
path_df_res = SAVE_RESULTS_PATH / 'results_em_synthetic.csv'
df_res.to_pickle(path_df_res)

# ===================================================
# PLOT FIGURES
# ===================================================

start, stop = 0, 1
xx = np.linspace(start, stop, (stop - start) * 300)

comb1 = {'sigma': 0.2, **simu_params}  # "wide" kernel scenario
comb2 = {'sigma': 0.05, **simu_params}  # "sharp" kernel scenario
combs = [comb1, comb2]

# ------ PLOT KERNEL RETRIEVAL ------
# (Figure 2 in paper)

T = 1000
n_tasks = 0.5
list_seeds = np.random.choice(list(range(50)), size=8, replace=False)

fig, axes = plt.subplots(1, 2, figsize=figsize)
for i, comb in enumerate(combs):
    # filter resulsts dataframe
    df_comb = df_res[(df_res['T'] == T) &
                     (df_res['n_tasks'] == n_tasks) &
                     (df_res['lower'] == comb['lower']) &
                     (df_res['upper'] == comb['upper']) &
                     (df_res['baseline'] == comb['baseline']) &
                     (df_res['alpha'] == comb['alpha']) &
                     (df_res['m'] == comb['m']) &
                     (df_res['sigma'] == comb['sigma'])]
    # check if no warning wera raised
    if min(df_comb['C'].values) == 0 or \
            min(df_comb['C_sigma'].values) == 0:
        print('C or C_sigma is 0')
    # define true kernel
    kernel_true = TruncNormKernel(
        comb['lower'], comb['upper'], comb['m'], comb['sigma'])

    yy = comb['baseline'] + kernel_true.eval(xx)
    axes[i].plot(xx, yy, label='Ground truth', color='black', linestyle='--')

    # define estimated kernels
    for j, seed in enumerate(list_seeds):
        sub_df = df_comb[df_comb['seed'] == seed]
        kernel = TruncNormKernel(comb['lower'], comb['upper'],
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
    title = rf"$\mu={comb['baseline']}$, "
    title += rf" $\alpha={comb['alpha']}$, "
    title += rf" $m={comb['m']}$, "
    title += rf" $\sigma={comb['sigma']}$"
    axes[i].legend(fontsize=fontsize)
    axes[i].set_title(title, fontsize=fontsize)

plt.tight_layout()

name = 'fig2.pdf'
path_fig = SAVE_RESULTS_PATH / name
plt.savefig(path_fig, dpi=300)
plt.savefig(str(path_fig).replace('pdf', 'png'), dpi=300)
plt.show()
plt.close()

# ------ PLOT MEAN/STD OF THE RELATIVE NORM ------
# (Figure 3 and A.1 in paper)


# plot mean
fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=figsize)
