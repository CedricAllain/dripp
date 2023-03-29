# %%
from sklearn import preprocessing, cluster
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

import mne
import mne_bids
from alphacsc.datasets import somato

from dripp.experiments.run_cdl import run_cdl_somato


u_hat_ = np.load('./u_hat_somato.npy')

_, info = somato.load_data(sfreq=150.)

fname_bem = './somato-5120-bem-sol.fif'
data_path = mne.datasets.somato.data_path()
raw_path = mne_bids.BIDSPath(subject='01', root=data_path, datatype='meg',
                             extension='.fif', task='somato')
trans = mne_bids.get_head_mri_trans(raw_path)

evoked = mne.EvokedArray(u_hat_.T, info)
dip = mne.fit_dipole(evoked, info['cov'], fname_bem, trans,
                     n_jobs=6, verbose=False)[0]

# %% get positions coordinates and save as csv
df_dip_pos = pd.DataFrame(data=dip.pos, columns=['x', 'y', 'z'])
df_dip_pos['atom_id'] = range(dip.pos.shape[0])
df_dip_pos.to_csv('df_dip_pos_somato.csv', index=False)

# compute dendograme
data = np.array(df_dip_pos[['x', 'y', 'z']])
data = preprocessing.StandardScaler().fit_tr(data)
plt.title("CAH on somato dipole fit")
dendrogram(linkage(data, method='ward'), labels=df_dip_pos.index,
           orientation='right')
plt.show()

n_clusters = 4

# %% Recompute CAH clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                     linkage='ward')
clustering.fit(data)
labels = clustering.labels_

# %% With k-means
# kmeans = cluster.KMeans(n_clusters=n_clusters)
# kmeans.fit(data)
# labels = kmeans.labels_

# %% plot topomap for one class

def plot_topomaps(atom_idx, u_hat_, info):

    n_columns = min(4, len(atom_idx))
    split = int(np.ceil(len(atom_idx) / n_columns))
    figsize = (4 * n_columns, 3 * split)
    fig, axes = plt.subplots(split, n_columns, figsize=figsize)
    axes = np.atleast_2d(axes)

    for ii, kk in enumerate(atom_idx):
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row:(i_row + 1), i_col])

        ax = next(it_axes)
        ax.set_title('Atom % d' % kk, pad=0)

        mne.viz.plot_topomap(data=u_hat_[kk], pos=info, axes=ax, show=False)
    
    fig.tight_layout()
    plt.show()


for this_label in range(clustering.n_clusters):
    print('Label %i' % this_label)
    atom_idx = np.where(labels == this_label)[0]
    plot_topomaps(atom_idx, u_hat_, info)


# %%
