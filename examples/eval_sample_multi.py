# %%

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel


N_JOBS = 10  # number of jobs to run in parallel. To adjust based on machine

# CDL parameters
cdl_params = {
    'n_atoms': 40,
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'reg': 0.1,
    'n_jobs': 5,
    'n_splits': 10
}
# run CDL and EM
lower, upper = 30e-3, 500e-3
shift_acti = True
threshold = 0.6e-10
n_iter = 400
dict_global, df_res = run_multiple_em_on_cdl(
    data_source='sample', cdl_params=cdl_params,  # CDL
    shift_acti=shift_acti, atom_to_filter='all', threshold=threshold,
    list_atoms=list(range(cdl_params['n_atoms'])),
    list_tasks=([1, 2], [3, 4]),  # auditory and visual stimuli, in two drivers
    n_driver=2,
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',  # EM
    n_jobs=N_JOBS)

# get raw.info
data_utils = utils.get_data_utils(data_source='sample', verbose=False)
raw = mne.io.read_raw_fif(data_utils['file_name'])
raw.pick_types(meg='grad', eeg=False, eog=False, stim=True)
info = raw.copy().pick_types(meg=True).info

# %%
# ==================================================================
# PLOT A SELECTION OF ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ==================================================================

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
})

# atoms and tasks to plot
plotted_atoms = [0, 1, 2, 6]
plotted_tasks = {'auditory': [1, 2],
                 'visual': [3, 4]}

fig = plt.figure(figsize=(5.5, 3.5))
gs = gridspec.GridSpec(nrows=3, ncols=4, hspace=0.26, wspace=0.18, figure=fig)

# x axis for temporal pattern
n_times_atom = dict_global['dict_cdl_params']['n_times_atom']
t = np.arange(n_times_atom) / cdl_params['sfreq']
# x axis for estimated intensity function
xx = np.linspace(0, 500e-3, 500)

u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])

for ii, kk in enumerate(plotted_atoms):
    # Select the current atom
    u_k = u_hat_[kk]
    v_k = v_hat_[kk]

    # Plot the spatial map of the atom using mne topomap
    ax = fig.add_subplot(gs[0, ii])
    mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
    ax.set_title('Atom % d' % kk, fontsize=fontsize, pad=0)
    if ii == 0:
        ax.set_ylabel('Spatial', labelpad=28, fontsize=fontsize)

    # Plot the temporal pattern of the atom
    ax = fig.add_subplot(gs[1, ii])
    if kk != 0:
        v_k = -1 * np.array(v_k)
    ax.plot(t, v_k)

    if ii == 0:
        temporal_ax = ax
        ax.set_ylabel('Temporal', fontsize=fontsize)

    if ii > 0:
        ax.get_yaxis().set_visible(False)
        temporal_ax.get_shared_y_axes().join(temporal_ax, ax)
        ax.autoscale()

    ax.set_xlim(0, 1)
    ax.set_xticklabels([0, 0.5, 1], fontsize=fontsize)

    # Plot the learned density kernel
    ax = fig.add_subplot(gs[2, ii])

    has_m_line = False
    df_temp = df_res[(df_res['atom'] == kk)
                     & (df_res['lower'] == lower)
                     & (df_res['upper'] == upper)
                     & (df_res['threshold'] == threshold)
                     & (df_res['shift_acti'] == shift_acti)]
    for jj, label in enumerate(plotted_tasks.keys()):
        # select sub-df of interest
        # in case that there has been an early stopping
        n_iter_temp = min(n_iter, df_temp['n_iter'].values.max())
        df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]
        # unpack parameters estimates
        alpha = list(df_temp['alpha_hat'])[0][jj]
        baseline = list(df_temp['baseline_hat'])[0]
        m = list(df_temp['m_hat'])[0][jj]
        sigma = list(df_temp['sigma_hat'])[0][jj]

        # define kernel function
        kernel = TruncNormKernel(lower, upper, m, sigma)
        yy = baseline + alpha * kernel.eval(xx)
        lambda_max = baseline + alpha * kernel.max
        ratio_lambda_max = lambda_max / baseline

        if ii > 0:
            plot_label = None
        else:
            plot_label = label

        ax.plot(xx, yy, label=plot_label)

        if (ratio_lambda_max > 1) and kk not in [0, 1]:
            has_m_line = True
            ax.vlines(m, ymin=0, ymax=lambda_max, color='black',
                      linestyle='--', label=r'%.3f' % m)

    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_xticklabels([0, 0.25, 0.5], fontsize=fontsize)

    if ii == 0:
        intensity_ax = ax
        ax.set_ylabel('Intensity', labelpad=7, fontsize=fontsize)
    else:
        ax.get_yaxis().set_visible(False)
        intensity_ax.get_shared_y_axes().join(intensity_ax, ax)
        ax.autoscale()

    ax.set_xlim(0, 500e-3)

    if plot_label is not None or has_m_line:
        ax.legend(fontsize=fontsize, handlelength=1)

# save figure
path_fig = SAVE_RESULTS_PATH / 'fig4_multi_bis.pdf'
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
plt.close()

# %%
