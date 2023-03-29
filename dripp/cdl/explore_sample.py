# %%
from collections import Counter
from alphacsc.datasets.mne_data import load_data

X, info = load_data(dataset='sample')
print(Counter(info['temp']['events'][:, 2]))
# %%
