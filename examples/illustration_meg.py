# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

import mne

from dripp.cdl import utils
from dripp.experiments.run_cdl import run_cdl_sample
# recover MNE sample atoms

dict_global = run_cdl_sample(sfreq=150., n_atoms=40, n_times_atom=None, reg=0.1,
                             n_iter=100, eps=1e-4, n_jobs=5, n_splits=10)
# pickle.dump(dict_global, open('dict_global_sample.pkl', "wb"))


# %%

u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])  # (n_atoms, 203)
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])
# z_hat = np.array(dict_global['dict_cdl_fit_res']['z_hat'])
pickle.dump(u_hat_, open('u_hat_sample.pkl', "wb"))
pickle.dump(v_hat_, open('v_hat_sample.pkl', "wb"))

plotted_atoms = [0, 1, 2, 6]

ecg = v_hat_[0]
ecg_spa = u_hat_[0]
eog = v_hat_[1]
eog_spa = u_hat_[1]
aud = v_hat_[2]
aud_spa = u_hat_[2]
vis = v_hat_[6]
vis_spa = u_hat_[6]

# %%
n_sensors = 7
T = 6  # seconds
sfreq = 150.
fig_high = 5.8

# %%
rng = np.random.RandomState(42)

tt_aud = [100, 400, 750]
aud_jitters = rng.normal(-0.085, 0.1, len(tt_aud))
tt_aud_jitt = (tt_aud + aud_jitters * sfreq).astype(int)
z_aud = np.zeros(int(T*sfreq))
z_aud[tt_aud] = 1
aud_sig = np.convolve(z_aud, aud, mode='full')[:int(T*sfreq)]

tt_vis = [250, 600]
vis_jitters = rng.normal(-0.188, 0.1, len(tt_vis))
tt_vis_jitt = (tt_vis + vis_jitters * sfreq).astype(int)
z_vis = np.zeros(int(T*sfreq))
z_vis[tt_vis] = 1
vis_sig = np.convolve(z_vis, vis, mode='full')[:int(T*sfreq)]

tt_ecg = [10, 200, 380, 570, 780]
z_ecg = np.zeros(int(T*sfreq))
z_ecg[tt_ecg] = 1
ecg_sig = np.convolve(z_ecg, ecg, mode='full')[:int(T*sfreq)]

tt_eog = [30, 250, 500, 700]
z_eog = np.zeros(int(T*sfreq))
z_eog[tt_eog] = 1
ocg_sig = np.convolve(z_eog, eog, mode='full')[:int(T*sfreq)]

list_sig = [aud_sig, vis_sig, ecg_sig, ocg_sig]
y_min = min([sig.min() for sig in list_sig])
y_max = max([sig.max() for sig in list_sig])

list_tt = [tt_aud, tt_vis, tt_ecg, tt_eog]

# %%
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True,
                         figsize=(6.4, fig_high))
for i, sig, tt in zip(range(4), list_sig, list_tt):
    axes[i].plot(sig)
    # for t in tt:
        # axes[i].axvline(t, y_min, y_max, linestyle='--', color='k')
    markerline, stemlines, baseline = axes[i].stem(
        tt, np.ones(len(tt))*0.25, bottom=y_min)
    plt.setp(stemlines, 'color', 'k', linewidth=1, linestyle='--')
    plt.setp(markerline, 'color', 'k', linewidth=2)
    plt.setp(baseline, visible=False)
    plt.setp(baseline, 'linewidth', 0)
    axes[i].set_xlim(0, int(T*sfreq))

axes[-1].set_xlabel("Time (s.)")
fig.tight_layout()
plt.savefig('./illustration_cdl.pdf')
plt.savefig('./illustration_cdl.png')
plt.show()

# %%
fig, axes = plt.subplots(4, sharex=True, sharey=True, figsize=(2, fig_high))
for i, atom in enumerate([aud, vis, ecg, eog]):
    axes[i].plot(atom)
    axes[i].set_xlim(0, 150)

axes[-1].set_xlabel("Time (s.)")
fig.tight_layout()
plt.savefig('./illustration_atom.pdf')
plt.savefig('./illustration_atom.png')
plt.show()

# %%
# aud_coef = np.array([0.01, 0.1,  1,    0.05,  0.9,    0.28])
# vis_coef = np.array([0.1,  0.01, 0.03, 1,     0.1,    0.9])
# ecg_coef = np.array([1,    0.2,  0.1,  0.1,   0.25,   0.4])
# ocg_coef = np.array([0.05, 1,    0.08, 0.12,  0.31,   0.2])

coefs = np.append(np.linspace(0.1, 1, n_sensors+1)[:-2], 1)
rng.shuffle(coefs)
aud_coef = np.roll(coefs, 0)
vis_coef = np.roll(coefs, 2)
ecg_coef = np.roll(coefs, 4)
ocg_coef = np.roll(coefs, 6)

X = rng.normal(0, 0.01, (n_sensors, int(T*sfreq)))
# add aud
X += np.outer(aud_coef, aud_sig)
# add vis
X += np.outer(vis_coef, vis_sig)
# add ecg
X += np.outer(ecg_coef, ecg_sig)
# add eog
X += np.outer(ocg_coef, ocg_sig)


fig, axes = plt.subplots(n_sensors+1, 1, sharex=True, sharey=True,
                         figsize=(6.4, fig_high))
for i in range(n_sensors):
    axes[i].plot(X[i])
# add stem
for tt, color in zip([tt_aud_jitt, tt_vis_jitt], ['blue', 'green']):
    markerline, stemlines, baseline = axes[-1].stem(tt, np.ones(len(tt)))
    plt.setp(stemlines, 'color', color, linewidth=1, linestyle='--')
    plt.setp(markerline, 'color', color, linewidth=2)
    plt.setp(baseline, visible=False)
    plt.setp(baseline, 'linewidth', 0)

axes[-1]._shared_y_axes.remove(axes[-1])
axes[-1].set_xlim(0, int(T*sfreq))
axes[-1].set_xlabel("Time (s.)")
fig.tight_layout()
plt.savefig('./illustration_meg.pdf')
plt.savefig('./illustration_meg.png')
plt.show()

# %%
