# %%
import mne
from mne.preprocessing import ICA
from mne.datasets import sample

# Load the MNE sample dataset
data_path = sample.data_path()
print(f"data_ath: {data_path}")
sample_data_raw_file = data_path / "MEG/sample/sample_audvis_filt-0-40_raw.fif"

# Read the raw data file
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True)
raw.load_data()

# pick some channels that clearly show heartbeats and blinks
# regexp = r"(MEG [12][45]31|EEG 00[1234])"
# artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
# raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# filtering to remove slow drifts
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)

# raw.load_data()
# fig = ica.plot_sources(raw, show_scrollbars=False)
# ica.plot_components(picks=range(5), ncols=1)

# # Choose the MEG channels
# picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)

# # Define the ICA object
# ica = ICA(n_components=20, random_state=0)

# # Fit the ICA model to the MEG data
# ica.fit(raw, picks=picks_meg)

# # Plot the independent components
# ica.plot_components(picks=range(3))  # Plot the first 3 components

# # Plot the time course of the first 3 components
# ica.plot_sources(raw, picks=range(3))


from PIL import Image

# Plot the components and save the figure
fig_comp = ica.plot_components(picks=range(5), colorbar=True, ncols=1)
fig_comp.savefig("components.png")

# Plot the sources and save the figure
fig_src = ica.plot_sources(raw, picks=range(5), show_scrollbars=False)
fig_src.savefig("sources.png")

# Open the saved images
img_comp = Image.open("components.png")
img_src = Image.open("sources.png")

# Create a new image with the size to accommodate both figures
img = Image.new(
    "RGB", (img_comp.width + img_src.width, max(img_comp.height, img_src.height))
)

# Paste the source image on the left
img.paste(img_src, (0, 0))

# Paste the component image on the right
img.paste(img_comp, (img_src.width, 0))

# Save the combined image
img.save("combined.png")
# %%
