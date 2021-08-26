"""
.. _tut-visualize-evoked:

Visualizing Evoked data
=======================

This tutorial shows the different visualization methods for
`~mne.Evoked` objects.

As usual we'll start by importing the modules we need:

Source:
- https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html
- https://mne.tools/stable/auto_tutorials/epochs/20_visualize_epochs.html
- https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html

"""
# %%
import os
import numpy as np
import mne

from dripp.config import SAVE_RESULTS_PATH

###############################################################################
# Instead of creating the `~mne.Evoked` object from an `~mne.Epochs` object,
# we'll load an existing `~mne.Evoked` object from disk. Remember, the
# :file:`.fif` format can store multiple `~mne.Evoked` objects, so we'll end up
# with a `list` of `~mne.Evoked` objects after loading. Recall also from the
# :ref:`tut-section-load-evk` section of :ref:`the introductory Evoked tutorial
# <tut-evoked-class>` that the sample `~mne.Evoked` objects have not been
# baseline-corrected and have unapplied projectors, so we'll take care of that
# when loading:

sample_data_folder = mne.datasets.sample.data_path()
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)

# Show the condition names, and reassure ourselves that baseline correction has
# been applied.
for e in evokeds_list:
    print(f'Condition: {e.comment}, baseline: {e.baseline}')
###############################################################################
# To make our life easier, let's convert that list of `~mne.Evoked`
# objects into a :class:`dictionary <dict>`. We'll use ``/``-separated
# dictionary keys to encode the conditions (like is often done when epoching)
# because some of the plotting methods can take advantage of that style of
# coding.

conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))
#      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ this is equivalent to:
# {'aud/left': evokeds_list[0], 'aud/right': evokeds_list[1],
#  'vis/left': evokeds_list[2], 'vis/right': evokeds_list[3]}

# %%

for this_cond, this_evk in evks.items():
    fig = this_evk.plot_joint(picks='mag')
    fig.suptitle('Epochs for %s stimulus' % this_cond, fontsize=11)
    fig_name = 'epoch_joint_%s_sample.pdf' % this_cond.replace('/', '-')
    fig.savefig(SAVE_RESULTS_PATH / fig_name)
    fig.savefig(SAVE_RESULTS_PATH / fig_name.replace('pdf', 'png'))

fig = mne.viz.plot_compare_evokeds(evks, picks='mag',
                                   colors=dict(aud=0, vis=1),
                                   linestyles=dict(left='solid',
                                                   right='dashed'))[0]
fig.savefig(SAVE_RESULTS_PATH / 'compare_evokeds_mag_sample.pdf')
fig.savefig(SAVE_RESULTS_PATH / 'compare_evokeds_mag_sample.png')

# plot frequency with channels on y axis and time on x axis
evks['vis/right'].plot_image(picks='meg')


###############################################################################
# %%
# =============================================
# Visualize epochs
# =============================================

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=120)

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, event_id=event_dict,
                    preload=True)

ecg_proj_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                             'sample_audvis_ecg-proj.fif')
ecg_projs = mne.read_proj(ecg_proj_file)
epochs.add_proj(ecg_projs)
epochs.apply_proj()

stimuli_types = ['auditory/left', 'auditory/right', 'visual/left',
                 'visual/right', 'auditory', 'visual']
for this_stimulus in stimuli_types:
    fig = epochs[this_stimulus].plot_image(picks='mag', combine='mean')[0]
    fig.suptitle('Epochs for %s stimulus' % this_stimulus, fontsize=11)
    fig_name = 'epoch_%s_sample.pdf' % this_cond.replace('/', '-')
    fig.savefig(SAVE_RESULTS_PATH / fig_name)
    fig.savefig(SAVE_RESULTS_PATH / fig_name.replace('pdf', 'png'))

###############################################################################
# %%
# =============================================
# ICA
# =============================================
