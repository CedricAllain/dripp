"""
Plot the topographies of CSC and ICA ECG/EOG artifacts, for sample dataset
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet

from dripp.config import SAVE_RESULTS_PATH


SAVE_RESULTS_PATH /= 'usual_methods_sample'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
# %%
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60.)

# Plot EOG (eye-blink) artifact
# eog_evoked = create_eog_epochs(raw).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# figs = eog_evoked.plot_joint()
# for this_fig, types in zip(figs, ['meg', 'eeg', 'grad']):
#     this_fig.savefig(SAVE_RESULTS_PATH /
#                      ('eog_evoked_%s.pdf' % types), dpi=300)

# Plot ECG (heartbeat) artifact
# ecg_evoked = create_ecg_epochs(raw).average()
# ecg_evoked.apply_baseline(baseline=(None, -0.2))
# figs = ecg_evoked.plot_joint()
# for this_fig, types in zip(figs, ['meg', 'eeg', 'grad']):
#     this_fig.savefig(SAVE_RESULTS_PATH /
#                      ('ecg_evoked_%s.pdf' % types), dpi=300)

# Compute and plot ICA
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(filt_raw)
raw.load_data()

fig = ica.plot_sources(raw, picks=[0, 1], show_scrollbars=False)
fig.savefig(SAVE_RESULTS_PATH / 'ica_sources.pdf', dpi=300)
fig = ica.plot_components(picks=[0, 1], ch_type='grad')
fig.savefig(SAVE_RESULTS_PATH / 'ica_components.pdf', dpi=300)
# figs = ica.plot_properties(raw, picks=[0, 1])  # 0: blinks, 1: heartbeats
# for this_fig, art in zip(figs, ['blinks', 'heartbeats']):
#     this_fig.savefig(SAVE_RESULTS_PATH /
#                      ('ica_properties_%s.pdf' % art), dpi=300)

# plot an overlay of the original signal against the reconstructed signal
# with the artifactual ICs excluded
# fig = ica.plot_overlay(raw, exclude=[0], picks='eeg')  # blinks
# fig.savefig(SAVE_RESULTS_PATH / 'ica_overlay_blinks.pdf', dpi=300)
# fig = ica.plot_overlay(raw, exclude=[1], picks='mag')  # heartbeats
# fig.savefig(SAVE_RESULTS_PATH / 'ica_overlay_heartbeats.pdf', dpi=300)

# reconstruct the sensor signals with artifacts removed
ica.exclude = [0, 1]  # indices chosen based on various plots above
reconst_raw = raw.copy()
ica.apply(reconst_raw)

regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
               show_scrollbars=False)
fig.savefig(SAVE_RESULTS_PATH / 'raw_artifact_picks.pdf', dpi=300)
fig = reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                       show_scrollbars=False)
fig.savefig(SAVE_RESULTS_PATH / 'reconst_raw_artifact_picks.pdf', dpi=300)
del reconst_raw

# %% Plot evoked data
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))
for this_cond in ['aud', 'vis']:
    evoked = mne.combine_evoked(
        [evks[this_cond + '/left'], evks[this_cond + '/right']],
        weights='nave')
    figs = evoked.plot_joint()
    figs[1].savefig(SAVE_RESULTS_PATH /
                    (this_cond + '_evoked_joint.pdf'), dpi=300)

# def custom_func(x):
#     return x.max(axis=1)


# # plot compared evokeds
# for combine in ('mean', 'median', 'gfp', custom_func):
#     mne.viz.plot_compare_evokeds(evks, picks='grad', combine=combine)


# %% Time frequency plan
# events = mne.find_events(raw, stim_channel='STI 014')
# event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
#               'visual/right': 4, 'face': 5, 'button': 32}
# reject_criteria = dict(mag=4000e-15,     # 4000 fT
#                        grad=4000e-13,    # 4000 fT/cm
#                        eeg=150e-6,       # 150 µV
#                        eog=250e-6)       # 250 µV
# epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
#                     reject=reject_criteria, preload=True)
# conds_we_care_about = ['auditory/left', 'auditory/right',
#                        'visual/left', 'visual/right']
# epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place

# aud_epochs = epochs['auditory']
# frequencies = np.arange(7, 30, 3)
# power = tfr_morlet(aud_epochs, n_cycles=2, return_itc=False, freqs=frequencies,
#                    decim=3)
# power.plot(['MEG 1332'])

# figs = power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
#                         timefreqs=[(.1, 7)])
# figs[-1].savefig(SAVE_RESULTS_PATH / ('aud_epochs_time_freq.pdf'), dpi=300)

# vis_epochs = epochs['visual']
# frequencies = np.arange(7, 30, 3)
# power = tfr_morlet(vis_epochs, n_cycles=2, return_itc=False, freqs=frequencies,
#                    decim=3)
# figs = power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
#                         timefreqs=[(.19, 7)])
# figs[-1].savefig(SAVE_RESULTS_PATH / ('vis_epochs_time_freq.pdf'), dpi=300)
