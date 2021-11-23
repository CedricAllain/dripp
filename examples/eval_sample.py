# %%

import os
import numpy as np
import mne
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from alphacsc.utils.convolution import construct_X_multi

from dripp.experiments.run_multiple_em_on_cdl import \
    run_multiple_em_on_cdl
from dripp.cdl import utils
from dripp.config import SAVE_RESULTS_PATH, N_JOBS
from dripp.trunc_norm_kernel.model import TruncNormKernel
from dripp.experiments.utils_plot import plot_cdl_atoms

SAVE_RESULTS_PATH /= 'results_sample'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

# %%

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

# %% get raw
data_utils = utils.get_data_utils(data_source='sample', verbose=False)
raw = mne.io.read_raw_fif(data_utils['file_name'], preload=True)
raw.pick_types(meg='grad', eeg=False, eog=True, stim=True)
raw.notch_filter(np.arange(60, 181, 60))
raw.filter(l_freq=2, h_freq=None)

# get epoch
events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'face': 5, 'button': 32}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, event_id=event_id,
                    preload=True)


# get info only for MEG
info = raw.copy().pick_types(meg=True).info

# %% Run CSC

dict_global, df_res = run_multiple_em_on_cdl(
    data_source='sample', cdl_params=cdl_params,  # CDL
    shift_acti=shift_acti, atom_to_filter='all', threshold=threshold,
    list_atoms=list(range(cdl_params['n_atoms'])),
    list_tasks=([1, 2], [3, 4]),  # auditory and visual stimuli, in two drivers
    n_drivers=2,
    lower=lower, upper=upper, n_iter=n_iter, initializer='smart_start',  # EM
    n_jobs=N_JOBS)

u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])  # (n_atoms, 203)
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])
z_hat = np.array(dict_global['dict_cdl_fit_res']['z_hat'])

# %%
# ==================================================================
# PLOT THE TOP 5 ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ==================================================================
# tasks to plot
plotted_tasks = {'auditory': [1, 2],
                 'visual': [3, 4]}

fig_name = 'fig4_top_5_atoms.pdf'
plot_cdl_atoms(dict_global, cdl_params, info,
               n_top_atoms=5, plot_intensity=True,
               alpha_threshold=-100,
               df_res_dripp=df_res, plotted_tasks=plotted_tasks,
               save_fig=True, path_fig=SAVE_RESULTS_PATH / fig_name)

# %%
# ==================================================================
# PLOT A SELECTION OF ATOMS AND THEIR ESTIMATED INTENSITY FUNCTIONS
# ==================================================================
# atoms to plot
plotted_atoms = [0, 1, 2, 6]

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
})

fig = plt.figure(figsize=(5.5, 3.5))
gs = gridspec.GridSpec(nrows=3, ncols=4, hspace=0.26, wspace=0.18, figure=fig)

# x axis for temporal pattern
n_times_atom = dict_global['dict_cdl_params']['n_times_atom']
t = np.arange(n_times_atom) / cdl_params['sfreq']
# x axis for estimated intensity function
xx = np.linspace(0, 500e-3, 500)

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
        # ratio_lambda_max = lambda_max / baseline
        ratio_lambda_max = alpha / baseline

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
plt.show()
plt.close()

# %% Compare to classical method of evoked responses

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis-ave.fif')
# evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
#                                 proj=True, verbose=False)
# conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
# evks = dict(zip(conds, evokeds_list))

# for this_cond, v_k in zip(['aud', 'vis'], [v_hat_[2], v_hat_[6]]):
#     evoked = mne.combine_evoked(
#         [evks[this_cond + '/left'], evks[this_cond + '/right']],
#         weights='nave')
#     figs = evoked.plot_joint()
#     fig = figs[1]
#     ax = fig.axes[0].twinx()
#     ax.plot(t, v_k)
#     figs[1].savefig(SAVE_RESULTS_PATH /
#                     (this_cond + '_evoked_joint.pdf'), dpi=300)


for this_cond, v_k in zip(['auditory', 'visual'], [v_hat_[2], v_hat_[6]]):
    evoked = epochs[this_cond].average()
    figs = evoked.plot_joint()
    fig = figs[1]
    ax = fig.axes[0].twinx()
    ax.plot(t, v_k)
    figs[1].savefig(SAVE_RESULTS_PATH /
                    (this_cond + '_evoked_joint.pdf'), dpi=300)


# %% Compute cosine similarity between artifacts from CDL and from ICA

# Compute and plot ICA
ica = mne.preprocessing.ICA(n_components=40, max_iter='auto', random_state=97)
ica.fit(raw)
ica_sources = ica.get_sources(raw)

# %% ==================================================================
# COMPUTE THE COSINE SIMILARITY WITH SELECTED COMPONENTS
# ==================================================================
ica_components = ica.get_components()  # shape (203, 15)
kk = [0, 1, 2, 6]
u = u_hat_[kk]  # shape (kk, 203)
# compute L2 norm
u_2, ica_components_2 = np.linalg.norm(
    u, axis=1), np.linalg.norm(ica_components, axis=0)
# compute cosine similarity between CDL and ICA components
cosine_matrix = np.dot(u, ica_components) / np.outer(u_2, ica_components_2)
# get ICA component id that gives maximum similarity
id_ica = np.argmax(cosine_matrix, axis=1)

# %% plot pairs
fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
})
fig = plt.figure(figsize=(5.5, 3.5))
gs = gridspec.GridSpec(nrows=2, ncols=len(
    kk), hspace=0, wspace=0.1, figure=fig)
for i, k in enumerate(kk):
    ax = fig.add_subplot(gs[0, i])
    mne.viz.plot_topomap(u_hat_[k], info, axes=ax, show=False)
    ax.set_title('Atom % d' % k, fontsize=fontsize, pad=0)
    # if i == 0:
    #     ax.set_ylabel('Spatial', labelpad=28, fontsize=fontsize)

    ax = fig.add_subplot(gs[1, i])
    mne.viz.plot_topomap(
        ica_components[:, id_ica[i]], info, axes=ax, show=False)
    ax.set_title('ICA%i' % id_ica[i], fontsize=fontsize, pad=0)
    ax.set_xlabel("%.2f%%" % (
        cosine_matrix.max(axis=1)[i] * 100), fontsize=fontsize)

fig.savefig(SAVE_RESULTS_PATH / 'ica_components_max_cosine.pdf', dpi=300)
plt.show()

# %% Plot and save some figures
fig = ica.plot_sources(raw, picks=id_ica, show_scrollbars=False)
fig.savefig(SAVE_RESULTS_PATH / 'ica_sources.pdf', dpi=300)
fig = ica.plot_components(picks=id_ica, ch_type='grad')
fig.savefig(SAVE_RESULTS_PATH / 'ica_components.pdf', dpi=300)
# %%
# ==================================================================
# COMPUTE THE SIMILARITY BETWEEN ICA AND RECONSTRUCTED SIGNAL
# ==================================================================

# for i, k in enumerate(kk):
#     v_k = v_hat_[k]
#     v_k_1 = np.r_[[1], v_k][None]
#     z_k = z_hat[:, k:k + 1]
#     X_k = construct_X_multi(z_k, v_k_1, n_channels=1)[0, 0]
#     ica_source_k = ica_sources.get_data()[id_ica[i]]
#     lenght_min = min(len(X_k), len(ica_source_k))
#     cos_simi_source_k = 1 - cosine(X_k[:lenght_min], ica_source_k[:lenght_min])
#     print("cosine similarity between ICA source %i and reconstructed signal "
#           "for atom %i : %.2f%%" % (id_ica[i], k, cos_simi_source_k * 100))


# ==================================================================
# EVOKED FOR ICA FOR AUDITORY STIMULUS
# ==================================================================
k = 6
id = id_ica[np.where(np.array(kk) == k)[0][0]]
print('ICA id:', id)

# %%
events = mne.find_events(raw)  # Event IDs: [ 1  2  3  4  5 32]
# events[:, 0] -= raw.first_samp
# event_id = {'visual/left': 3, 'visual/right': 4}

# raw_from_ica = mne.io.RawArray(
#     np.atleast_2d(ica_sources.get_data()[id]),
#     mne.create_info(1, 150., ch_types='grad'))
ica_epochs = mne.Epochs(ica_sources,
                        events, event_id=event_id,
                        picks=[id],
                        tmin=-0.2, tmax=0.5, baseline=(None, 0.0),
                        preload=True)
fig = ica_epochs['visual'].plot_image()[0]
# %%

# get evoked from raw
sample_data_folder = mne.datasets.sample.data_path()
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))
this_cond = 'vis'
evoked_vis = mne.combine_evoked(
    [evks[this_cond + '/left'], evks[this_cond + '/right']], weights='nave').pick_types('grad')
# multiply by spatial atom k
evoked_raw = np.dot(np.atleast_2d(u_hat_[k]), evoked.data)

# reconstruct signal from CDL
v_k = v_hat_[k]
v_k_1 = np.r_[[1], v_k][None]
z_k = z_hat[:, k:k + 1]
X_k = construct_X_multi(z_k, v_k_1, n_channels=1)[0, 0]
# compute epoch
cdl_aud_epoch = mne.Epochs(mne.io.RawArray(np.atleast_2d(X_k),
                                           mne.create_info(1, 150., ch_types='grad')),
                           events, event_id=event_id,
                           tmin=-0.3, tmax=0.7, baseline=(None, 0.0),
                           preload=True)
cdl_aud_evk = cdl_aud_epoch.average().data[0]

# plot global figure
# ica.apply(epochs['visual'].average(), include=15).plot_joint(picks='grad')
fig = ica_aud_epoch.plot_image()[0]
plt.plot(evoked_raw[0], ax=fig.axes[1])
# figs = evoked.plot_joint()[1]
fig.savefig(SAVE_RESULTS_PATH / 'ica_vis_epoch.pdf', dpi=300)
plt.show()

# %%
