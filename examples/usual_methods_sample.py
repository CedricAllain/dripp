"""
Plot the topographies of CSC and ICA ECG/EOG artifacts, for sample dataset
"""
# %%
import os
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
# %%
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60.)

# Plot EOG (eye-blink) artifact
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

# Plot ECG (heartbeat) artifact
ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()

# Compute and plot ICA
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(filt_raw)
raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)
ica.plot_components()
ica.plot_properties(raw, picks=[0, 1])  # 0: blinks, 1: heartbeats

# plot an overlay of the original signal against the reconstructed signal
# with the artifactual ICs excluded
ica.plot_overlay(raw, exclude=[0], picks='eeg')  # blinks
ica.plot_overlay(raw, exclude=[1], picks='mag')  # heartbeats

# reconstruct the sensor signals with artifacts removed
ica.exclude = [0, 1]  # indices chosen based on various plots above
reconst_raw = raw.copy()
ica.apply(reconst_raw)

regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
         show_scrollbars=False)
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                 show_scrollbars=False)

# %% Plot evoked data
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))
for this_evoked in evokeds_list:
    this_evoked.plot_joint()


def custom_func(x):
    return x.max(axis=1)


# plot compared evokeds
for combine in ('mean', 'median', 'gfp', custom_func):
    mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)
    mne.viz.plot_compare_evokeds(evks, combine=combine)

# %%
