"""
Run EM on mne.somato dataset and plot the corresponding figure
(Figures 5, A.3, A.4 in paper)
"""

# %%
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
import mne_bids

from alphacsc.datasets import somato
from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
from dripp.config import SAVE_RESULTS_PATH, N_JOBS
from dripp.trunc_norm_kernel.model import TruncNormKernel
from dripp.experiments.utils_plot import plot_cdl_atoms

from mne.time_frequency import tfr_morlet


SAVE_RESULTS_PATH /= 'results_somato'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

cdl_params = {
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'n_jobs': 5,
    'n_splits': 10,
    'n_atoms': 20,
    'n_times_atom': 80,
    'reg': 0.2
}
# run CDL and EM
lower, upper = 0, 2
shift_acti = True
threshold = 1e-10
n_iter = 400
dict_global, df_res = run_multiple_em_on_cdl(
    data_source='somato', cdl_params=cdl_params,  # CDL
    shift_acti=shift_acti, atom_to_filter='all', threshold=threshold,
    list_atoms=list(range(cdl_params['n_atoms'])),
    list_tasks=[1], n_driver=1,
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',  # EM
    n_jobs=N_JOBS)

# %%

# get raw.info
sfreq = cdl_params['sfreq']
_, info = somato.load_data(sfreq=sfreq)

plotted_tasks = {'somatosensory': [1]}
fig_name = "somato_top_5_atoms.pdf"
plot_cdl_atoms(dict_global, cdl_params, info,
               n_top_atoms=5, plot_intensity=True,
               df_res_dripp=df_res, plotted_tasks=plotted_tasks,
               save_fig=True, path_fig=SAVE_RESULTS_PATH / fig_name)

# %%

# ==================================================================
# PLOT A SELECTION OF ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ==================================================================

# list of atoms selection to plot (3 graphes of 3 cherry picked atoms)
# plotted_atoms_list = [[2, 7, 10], [1, 2, 4], [0, 7, 10]]
plotted_atoms_list = [[0, 2, 7]]

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
    'legend.title_fontsize': fontsize
})

colors = ['blue', 'green', 'orange']

n_times_atom = cdl_params['n_times_atom']


u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])

# x axis for temporal pattern
t = np.arange(n_times_atom) / sfreq
# x axis for estimate intensity
xx = np.linspace(0, 2, 500)

for plotted_atoms in plotted_atoms_list:
    # define figure
    fig = plt.figure(figsize=(5.5, 3.5 / 3 * 2))
    ratio = 1.5  # ratio between width of atom plot and intensity plot
    step = 1/(3+ratio)
    gs = gridspec.GridSpec(nrows=2, ncols=4,
                           width_ratios=[step, step, step, ratio*step],
                           hspace=0.05,
                           wspace=0.1,
                           figure=fig)

    # plot spatial and temporal pattern
    for ii, kk in enumerate(plotted_atoms):
        # Select the current atom
        u_k = u_hat_[kk]
        v_k = v_hat_[kk]

        # plot spatial pattern
        ax = fig.add_subplot(gs[0, ii])
        ax.set_title('Atom % d' % kk, fontsize=fontsize)
        mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
        if ii == 0:
            ax.set_ylabel('Spatial', labelpad=32, fontsize=fontsize)

        # plot temporal pattern
        ax = fig.add_subplot(gs[1, ii])

        if kk == 0:  # return atom 0
            v_k = -1 * np.array(v_k)

        ax.plot(t, v_k, color=colors[ii])
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        if ii == 0:
            first_ax = ax
            ax.set_ylabel('Temporal', fontsize=fontsize)
        else:
            ax.get_yaxis().set_visible(False)
            first_ax.get_shared_y_axes().join(first_ax, ax)
            ax.autoscale()

        ax.set_xlim(0, n_times_atom / sfreq)
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_xticklabels([0, 0.25, 0.5], fontsize=fontsize)

    # plot EM-learned intensities
    ax = fig.add_subplot(gs[:, -1:])
    ax.set_title('Intensity', fontsize=fontsize)
    for ii, kk in enumerate(plotted_atoms):
        # select sub-df of interest
        df_temp = df_res[(df_res['atom'] == kk)
                         & (df_res['lower'] == lower)
                         & (df_res['upper'] == upper)
                         & (df_res['threshold'] == threshold)
                         & (df_res['shift_acti'] == shift_acti)]

        # if we save several values for n_iter
        if df_temp.shape[0] != 1:
            # in case that there has been an early stopping
            n_iter_temp = min(
                n_iter, df_temp['n_iter'].values.max())
            df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]

        list_yy = []
        for i in df_temp.index:
            # unpack parameters estimates
            alpha = df_temp['alpha_hat'][i][0]
            baseline = df_temp['baseline_hat'][i]
            m = df_temp['m_hat'][i][0]
            sigma = df_temp['sigma_hat'][i][0]

            # define kernel function
            kernel = TruncNormKernel(lower, upper, m, sigma)
            yy = baseline + alpha * kernel.eval(xx)
            list_yy.append(yy)

        label = '% d' % kk
        ax.plot(xx, yy, label=label, color=colors[ii])

        ax.set_xlim(0, 2)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.yaxis.set_ticks_position("right")
        # ax.set_yscale('log')
        ax.legend(fontsize=fontsize, handlelength=1, title='Atom')

    # save figure
    suffix = 'atom'
    for kk in plotted_atoms:
        suffix += '_' + str(kk)
    name = 'fig5_' + suffix + '_bis.pdf'
    path_fig = SAVE_RESULTS_PATH / name
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.close()

# %%
# ==================================================================
# PLOT THE DIPOLE FIT FOR THE SELECTED ATOMS
# ==================================================================

fname_bem = './somato-5120-bem-sol.fif'
data_path = mne.datasets.somato.data_path()
subjects_dir = data_path + '/derivatives/freesurfer/subjects'
raw_path = mne_bids.BIDSPath(subject='01', root=data_path, datatype='meg',
                             extension='.fif', task='somato')
trans = mne_bids.get_head_mri_trans(raw_path)

evoked = mne.EvokedArray(u_hat_.T, info)
dip = mne.fit_dipole(evoked, info['cov'], fname_bem, trans,
                     n_jobs=6, verbose=False)[0]

# for each of the cherry picked atoms plotted upper
for plotted_atoms in plotted_atoms_list:
    # define figure
    width = 6.5
    height = 1.8
    figsize = (width, width * height/5.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=len(plotted_atoms),
                           wspace=0.02,
                           figure=fig)

    fig_name = 'dipole_fit_atom'
    for i, i_atom in enumerate(plotted_atoms):
        fig_name += '_' + str(i_atom)

        ax = fig.add_subplot(gs[0, i], projection='3d')
        dip.plot_locations(trans, '01', subjects_dir,
                           idx=i_atom, ax=ax, show_all=False)
        ax.set_title('Atom %i' % i_atom, fontsize=fontsize, pad=0)
        # remove all ticks and associated labels to have a clear figure
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.set_zlabel('')

    fig.suptitle('')
    fig.tight_layout()

    fig_name += '_bis.pdf'
    path_fig = SAVE_RESULTS_PATH / fig_name
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.close()

# %%

# ================================================================
# PLOT ALL EXTRACTED ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ================================================================

plotted_atoms = range(cdl_params['n_atoms'])

# number of plots by atom
n_plots = 3
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    u_k = u_hat_[kk]
    v_k = v_hat_[kk]

    # Plot the spatial map of the atom using mne topomap
    ax = next(it_axes)
    ax.set_title('Atom % d' % kk, fontsize=fontsize, pad=0)

    mne.viz.plot_topomap(data=u_k, pos=info, axes=ax, show=False)
    if i_col == 0:
        ax.set_ylabel('Spatial', labelpad=28, fontsize=fontsize)

    # Plot the temporal pattern of the atom
    ax = next(it_axes)
    if kk == 0:  # return atom 0
        v_k = -1 * np.array(v_k)
    ax.plot(t, v_k)
    ax.set_xlim(0, n_times_atom / sfreq)

    if i_col == 0:
        ax.set_ylabel('Temporal', fontsize=fontsize)

    # plot the estimate intensity function
    ax = next(it_axes)
    # select sub-df of interest
    df_temp = df_res[(df_res['atom'] == kk)
                     & (df_res['lower'] == lower)
                     & (df_res['upper'] == upper)
                     & (df_res['threshold'] == threshold)
                     & (df_res['shift_acti'] == shift_acti)]
    # in case that there has been an early stopping
    n_iter_temp = min(n_iter, df_temp['n_iter'].values.max())
    df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]
    # unpack parameters estimates
    alpha = list(df_temp['alpha_hat'])[0][0]
    baseline = list(df_temp['baseline_hat'])[0]
    m = list(df_temp['m_hat'])[0][0]
    sigma = list(df_temp['sigma_hat'])[0][0]

    # define kernel function
    kernel = TruncNormKernel(lower, upper, m, sigma)
    yy = baseline + alpha * kernel.eval(xx)
    lambda_max = baseline + alpha * kernel.max
    ratio_lambda_max = lambda_max / baseline

    ax.plot(xx, yy)
    ax.set_xlim(0, 2)
    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_yscale('log')

    if ii == 0:
        intensity_ax = ax
    else:
        intensity_ax.get_shared_y_axes().join(intensity_ax, ax)
        ax.autoscale()

    if i_col == 0:
        ax.set_ylabel('Intensity', labelpad=7, fontsize=fontsize)

# save figure
fig.tight_layout()
path_fig = SAVE_RESULTS_PATH / 'somato_all_atoms_bis.pdf'
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
plt.savefig(str(path_fig).replace('pdf', 'png'), dpi=300, bbox_inches='tight')
plt.close()


# %% Compare to usual time/frequency analysis
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD (3-clause)
data_path = mne.datasets.somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

# Construct Epochs
event_id, tmin, tmax = 1, -1., 3.
baseline = (None, 0)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
                    preload=True)

epochs.resample(200., npad='auto')  # resample to reduce computation time
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
figs = power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
                        timefreqs=[(1, 7), (1.3, 8)])
figs.savefig(SAVE_RESULTS_PATH / ('somato_time_freq.pdf'), dpi=300)
