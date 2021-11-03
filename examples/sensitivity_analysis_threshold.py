"""Compare the DriPP results obtain with manually determined value of threshold
with a percentage of activation selection.
Experiments are performed on MNE sample dataset, for the 4 main atoms of
interest: artifacts (heartbeat and eye-blinks), audio and visual responses.
"""
# %%
# %%
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt

from dripp.experiments.run_cdl import run_cdl_sample
from dripp.cdl import utils
from dripp.trunc_norm_kernel.optim import em_truncated_norm
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.metric import infinite_norm_intensity

from dripp.config import SAVE_RESULTS_PATH
SAVE_RESULTS_PATH /= 'sensitivity_analysis_threshold'
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)

# ====== Get CDL results ======
# CDL parameters
sfreq = 150.
cdl_params = {
    'n_atoms': 40,
    'sfreq': sfreq,
    'n_iter': 100,
    'eps': 1e-4,
    'reg': 0.1,
    'n_jobs': 5,
    'n_splits': 10
}
dict_global = run_cdl_sample(**cdl_params)
# Select atoms of interest
list_atoms = [0, 1, 2, 6]
tasks = ([1, 2], [3, 4])  # auditory and visual stimuli, in two drivers
# Get results of interest
dict_pair_up = dict_global['dict_pair_up']
T = dict_pair_up['T']  # duration of process
events_timestamps = dict_pair_up['events_timestamps']
acti = np.array(dict_pair_up['acti_shift'])

# ====== With current value of threshold ======
threshold = 0.6e-10
atoms_timestamps = utils.get_atoms_timestamps(acti=acti,
                                              sfreq=sfreq,
                                              threshold=threshold)


def proprocess_tasks(tasks):
    if isinstance(tasks, int):
        tt = np.sort(events_timestamps[tasks])
    elif isinstance(tasks, list):
        tt = np.r_[events_timestamps[tasks[0]]]
        for i in tasks[1:]:
            tt = np.r_[tt, events_timestamps[i]]
        tt = np.sort(tt)

    return tt


# process events timestamps
if isinstance(tasks, tuple):
    # in that case, multiple drivers
    tt = np.array([proprocess_tasks(task) for task in tasks])
else:
    tt = proprocess_tasks(tasks)

n_drivers = len(tt)

# define EM parameters
em_params = {
    'lower': 30e-3,
    'upper': 500e-3,
    'initializer': 'smart_start',
    'n_iter': 400
}


def procedure(atom, atoms_timestamps):
    # get the activations for this particular atom id
    aa = atoms_timestamps[atom]
    # run EM
    res_em = em_truncated_norm(
        acti_tt=aa,
        driver_tt=tt,
        lower=em_params['lower'],
        upper=em_params['upper'],
        T=T,
        initializer=em_params['initializer'],
        n_iter=em_params['n_iter'],
        disable_tqdm=True)
    # extract and save results
    baseline_hat, alpha_hat, m_hat, sigma_hat = res_em[0]
    row = {**em_params,
           'threshold': threshold,
           'per_atom': per_atom,
           'atom': int(atom),
           'baseline_hat': baseline_hat,
           'alpha_hat': alpha_hat,
           'm_hat': m_hat,
           'sigma_hat': sigma_hat,
           }

    if np.isnan(per_atom):
        row['infinite_norm_of_diff_rel_kernel_%i' % atom] = 0
    else:
        # define estimated kernels
        kernel = [TruncNormKernel(em_params['lower'], em_params['upper'],
                                  m_hat[i], sigma_hat[i])
                  for i in range(n_drivers)]
        # define estimated intensity
        intensity_hat = Intensity(baseline_hat, alpha_hat, kernel)
        # compute infinite relative norm
        inf_norm_rel = infinite_norm_intensity(
            dict_intensity_true[atom], intensity_hat)
        row['infinite_norm_of_diff_rel_kernel_%i' % atom] = inf_norm_rel

    return row


per_atom = np.nan
new_rows = Parallel(n_jobs=4, verbose=1)(
    delayed(procedure)(this_atom, atoms_timestamps) for this_atom in list_atoms)

df_res = pd.DataFrame()
for new_row in new_rows:
    df_res = df_res.append(new_row, ignore_index=True)

df_res.to_csv(SAVE_RESULTS_PATH / 'results.csv')

# define dict of "true" intensities
dict_intensity_true = {}
for kk in list_atoms:
    sub_df = df_res[df_res['atom'] == kk]
    # unpack parameters estimates
    baseline = list(sub_df['baseline_hat'])[0]
    alpha = list(sub_df['alpha_hat'])[0]
    m = list(sub_df['m_hat'])[0]
    sigma = list(sub_df['sigma_hat'])[0]
    # define kernels
    kernel = [TruncNormKernel(
        em_params['lower'], em_params['upper'], m[i], sigma[i])
        for i in range(n_drivers)]
    # define true intensity
    dict_intensity_true[kk] = Intensity(baseline, alpha, kernel)

# ====== With threshold value as a percentage  ======
list_threshold = [0, 10, 20, 30, 40, 50]

for per_atom in [True, False]:
    for threshold in list_threshold:
        atoms_timestamps = utils.get_atoms_timestamps(acti=acti,
                                                      sfreq=sfreq,
                                                      threshold=threshold,
                                                      percent=True,
                                                      per_atom=per_atom)
        new_rows = Parallel(n_jobs=4, verbose=1)(
            delayed(procedure)(this_atom, atoms_timestamps)
            for this_atom in list_atoms)

        for new_row in new_rows:
            df_res = df_res.append(new_row, ignore_index=True)

        df_res.to_csv(SAVE_RESULTS_PATH / 'results.csv')

# ====== Plot graph results ======
# %%
lower, upper = 30e-3, 500e-3
per_atom = False
plotted_tasks = {'auditory': [1, 2],
                 'visual': [3, 4]}
fontsize = 9
plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]
# })

# plot the norm with respect to the threshold value

c = np.arange(1, len(list_atoms) + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap='tab20')

styles = ['-', '--']
labels = list(plotted_tasks.keys())

fig, ax = plt.subplots(figsize=(5.5, 2))
for i, atom in enumerate(list_atoms):
    columns = ['threshold', 'infinite_norm_of_diff_rel_kernel_%i' % atom]
    sub_df = df_res[(df_res['atom'] == atom) &
                    (df_res['per_atom'] == per_atom) &
                    (df_res['lower'] == lower) &
                    (df_res['upper'] == upper)][columns]
    sub_df = sub_df.set_index('threshold')
    for j, task in enumerate(plotted_tasks.keys()):
        sub_df[task] = sub_df['infinite_norm_of_diff_rel_kernel_%i' % atom].apply(
            lambda x: x[j])
    # sub_df = sub_df.unstack().apply(pd.Series)
    # sub_df.index = sub_df.index.droplevel()
    if i == 0:
        legend = True
    else:
        legend = False
    sub_df[plotted_tasks.keys()].plot(
        ax=ax, c=cmap.to_rgba(i+1), style=styles, legend=legend)

plt.savefig(SAVE_RESULTS_PATH / ('inf_norm.pdf'), dpi=300)
plt.show()
plt.close()

# %%

columns = ['threshold', 'baseline_hat', 'alpha_hat', 'm_hat', 'sigma_hat']
start, stop = 0, 0.5
figsize = (5.5, 2 * len(list_atoms))
xx = np.linspace(start, stop, int(np.ceil((stop - start) * 300)))
# define colors
c = np.arange(1, len(list_threshold) + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

fig, axes = plt.subplots(len(list_atoms), n_drivers, figsize=figsize)
for i, atom in enumerate(list_atoms):
    # get true intensity
    intensity = dict_intensity_true[atom]
    # get EM results for all values of threshold
    df_atom = df_res[(df_res['atom'] == atom) &
                     (df_res['per_atom'] == per_atom) &
                     (df_res['lower'] == lower) &
                     (df_res['upper'] == upper)][columns]
    #
    axes[i, 0].set_ylabel('Intensity atom %i' % atom, fontsize=fontsize)
    for p in range(n_drivers):
        if i == 0:
            axes[i, p].set_title(list(plotted_tasks.keys())[p],
                                 fontsize=fontsize)
        if atom == list_atoms[-1]:
            axes[i, p].set_xlabel('T (s)')
        # plot 'true' kernel
        yy_true = intensity.baseline + \
            intensity.alpha[p] * intensity.kernel[p].eval(xx)
        axes[i, p].plot(xx, yy_true, color='black', linestyle='--')
        for j, threshold in enumerate(list_threshold):
            sub_df = df_atom[df_atom['threshold'] == threshold]
            # define estimated kernel
            kernel = TruncNormKernel(em_params['lower'], em_params['upper'],
                                     list(sub_df['m_hat'])[0][p],
                                     list(sub_df['sigma_hat'])[0][p])
            yy_hat = sub_df['baseline_hat'].values[0] + \
                sub_df['alpha_hat'].values[0][p] * kernel.eval(xx)
            axes[i, p].plot(xx, yy_hat, alpha=0.8, label=str(threshold),
                            c=cmap.to_rgba(j+1))
        if i > 0 or p > 0:
            axes[0, 0].get_shared_y_axes().join(axes[0, 0], axes[i, p])
            axes[i, p].autoscale()
        if p == (n_drivers-1):
            axes[i, p].legend(fontsize=fontsize)

plt.savefig(SAVE_RESULTS_PATH / ('kernel_retrieval.pdf'), dpi=300)
plt.show()
# %%
