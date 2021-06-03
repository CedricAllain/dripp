"""
Run EM on mne.somato dataset and plot the corresponding figure
(Figure 5, A.3, A.4 in paper)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
import mne_bids

from alphacsc.datasets import somato
from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
from dripp.config import SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.model import TruncNormKernel


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
    list_tasks=[1],
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',  # EM
    n_jobs=50)

# save df_res as csv
path_df_res = SAVE_RESULTS_PATH / 'results_em_somato.csv'
df_res.to_csv(path_df_res)

# ================================================================
# PLOT A SELECTION OF ATOMS AND THEIR ESTIMATED INTENSITY FUNCTION
# ================================================================

# list of atoms selection to plot (3 graphes of 3 cherry picked atoms)
plotted_atoms_list = [[2, 7, 10], [1, 2, 4], [0, 7, 10]]

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
    'legend.title_fontsize': fontsize
})

colors = ['blue', 'green', 'orange']

# get raw.info
sfreq = cdl_params['sfreq']
_, info = somato.load_data(sfreq=sfreq)

n_times_atom = cdl_params['n_times_atom']

# x axis for temporal pattern
t = np.arange(n_times_atom) / sfreq
# x axis for estimate intensity
xx = np.linspace(0, 2, 500)

for plotted_atoms in plotted_atoms_list:
    # define figure
    fig = plt.figure(figsize=(5.5, 3.5 / 3 * 2))  # , constrained_layout=True)
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
        u_k = dict_global['dict_cdl_fit_res']['u_hat_'][kk]
        v_k = dict_global['dict_cdl_fit_res']['v_hat_'][kk]

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
        ax.set_xlabel('Time (s)', fontsize=fontsize)  # , fontsize=fontsize)
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
        # ax.set_xticklabels([0, 0.5], fontsize=fontsize)

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
            alpha = df_temp['alpha_hat'][i]
            baseline = df_temp['baseline_hat'][i]
            m = df_temp['m_hat'][i]
            sigma = df_temp['sigma_hat'][i]

            # define kernel function
            kernel = TruncNormKernel(lower, upper, m, sigma)
            yy = baseline + alpha * kernel.eval(xx)
            list_yy.append(yy)

        label = '% d' % kk
        ax.plot(xx, yy, label=label, color=colors[ii])

        ax.set_xlim(0, 2)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.yaxis.set_ticks_position("right")
        ax.set_yscale('log')
        ax.legend(fontsize=fontsize, handlelength=1, title='Atom')

    # save figure
    suffix = 'atom'
    for kk in plotted_atoms:
        suffix += '_' + str(kk)
    name = 'fig5_' + suffix + '.pdf'
    path_fig = SAVE_RESULTS_PATH / name
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# add the fit dipole plot
fname_bem = './somato-5120-bem-sol.fif'
data_path = mne.datasets.somato.data_path()
subjects_dir = data_path + '/derivatives/freesurfer/subjects'
raw_path = mne_bids.BIDSPath(subject='01', root=data_path, datatype='meg',
                             extension='.fif', task='somato')
trans = mne_bids.get_head_mri_trans(raw_path)

u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])
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

    fig_name += '.pdf'
    path_fig = SAVE_RESULTS_PATH / fig_name
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')


# ================================================================
# PLOT ALL EXTRACTED ATOMS AND THEIR ESTIMATE INTENSITY
# ================================================================
