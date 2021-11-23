# %%
from scipy.stats import t as student
from scipy.stats import norm
from scipy.stats import fisher_exact
from scipy.stats import chisquare
from dripp.trunc_norm_kernel.optim import compute_lebesgue_support
from scipy.sparse import find
import numpy as np
import pandas as pd

from dripp.cdl import utils
from dripp.experiments.run_cdl import run_cdl_sample, run_cdl_somato,\
    run_cdl_camcan
from dripp.trunc_norm_kernel.utils import check_acti_tt, check_driver_tt
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity

data_source = 'sample'
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

# get CDL results
if data_source == 'sample':
    dict_global = run_cdl_sample(**cdl_params)
elif data_source == 'somato':
    dict_global = run_cdl_somato(**cdl_params)
elif data_source == 'camcan':
    dict_global = run_cdl_camcan(**cdl_params)

# process events and activation timestamps
n_atoms = dict_global['dict_cdl_params']['n_atoms']
list_atoms = list(range(n_atoms))

if data_source == 'sample':
    tasks = ([1, 2], [3, 4])
    tasks_type = ['aud', 'vis']
elif data_source == 'camcan':
    tasks = ([1, 2, 3, 5], [1, 2, 3, 6])
elif data_source == 'somato':
    tasks = [1]

# get general parameters
print("Preprocess CDL results")
dict_other_params = dict_global['dict_other_params']
sfreq = dict_other_params['sfreq']
dict_pair_up = dict_global['dict_pair_up']
T = dict_pair_up['T']

# get events timestamps
events_timestamps = dict_pair_up['events_timestamps']  # type = dict

# get activations vectors
if shift_acti:
    acti = np.array(dict_pair_up['acti_shift'])
else:
    acti = np.array(dict_pair_up['acti_not_shift'])

acti = utils.filter_activation(acti, sfreq=sfreq)
atoms_timestamps = utils.get_atoms_timestamps(acti=acti,
                                              sfreq=sfreq,
                                              threshold=0, percent=True,
                                              per_atom=False)


def proprocess_tasks(tasks):
    if isinstance(tasks, int):
        tt = np.sort(events_timestamps[tasks])
    elif isinstance(tasks, list):
        tt = np.r_[events_timestamps[tasks[0]]]
        for i in tasks[1:]:
            tt = np.r_[tt, events_timestamps[i]]
        tt = np.sort(tt)

    return tt


if isinstance(tasks, tuple):
    # in that case, multiple drivers
    tt = np.array([proprocess_tasks(task) for task in tasks])
else:
    tt = proprocess_tasks(tasks)

driver_tt = check_driver_tt(tt)
n_drivers = len(driver_tt)

# %% ====================================
# Perform conformity test
# =======================================


def comformity_test(x_bar, mu_0, S, n, verbose=False, alternative='greater'):
    """

    """
    u = (x_bar - mu_0) / (S / np.sqrt(n))
    if n > 30:
        # Normal
        if alternative == 'les':
            p_val = norm.cdf(u)
        elif alternative == 'greater':
            p_val = norm.sf(u)
        elif alternative == 'two-sided':
            p_val = 2 * norm.sf(np.abs(u))
        else:
            raise ValueError("alternative must be "
                             "'less', 'greater' or 'two-sided'")
    else:
        # Student
        df = n - 1
        if alternative == 'less':
            p_val = student.cdf(u, df)
        elif alternative == 'greater':
            p_val = student.sf(u, df)
        elif alternative == 'two-sided':
            p_val = 2 * student.sf(np.abs(u), df)
        else:
            raise ValueError("alternative must be "
                             "'less', 'greater' or 'two-sided'")

    if verbose:
        print('statistic = %.3f, p-value = %.3f' % (u, p_val))

    return u, p_val


# define results DataFrame
df_col = ['atom_id', 'atom_type']
df_col.extend(['p_val_%s' % this_type for this_type in tasks_type])
df_res = pd.DataFrame(columns=df_col)

atom_dict = {0: 'heartbeat', 1: 'eye-blink', 2: 'auditory', 6: 'visual'}

# apply test for a selection of atoms
for atom_id in [0, 1, 2, 6]:
    new_row = {'atom_id': atom_id, 'atom_type': atom_dict[atom_id]}
    acti_tt = check_acti_tt(atoms_timestamps[atom_id])

    # get statistic for activations in kernel supports
    n_in_support = []
    ave_in_support = []
    std_in_support = []
    freq_in_support = []

    for p in range(n_drivers):
        # get activations in kernel support
        acti_in_support = np.array(
            [len(acti_tt[(acti_tt >= t + lower) * (acti_tt <= t + upper)])
             for t in driver_tt[p]])
        n = acti_in_support.sum()
        n_in_support.append(acti_in_support.sum())
        acti_in_support = acti_in_support / (upper - lower)
        ave_in_support.append(acti_in_support.mean())
        std_in_support.append(acti_in_support.std(ddof=1))
        freq_in_support.append(n_in_support[p] / len(acti_tt))

    # get statistic for activation over the baseline
    all_tt = np.sort(np.hstack(driver_tt))
    # compute lebesgue measure of all driver supports
    s = compute_lebesgue_support(all_tt, lower, upper)
    # compute average number of activations that lend over baseline
    n_in_baseline = len(acti_tt) - np.array(n_in_support).sum()
    ave_in_baseline = n_in_baseline / (T - s)
    freq_in_baseline = n_in_baseline / len(acti_tt)

    for p, this_type in enumerate(tasks_type):
        u, p_val = comformity_test(x_bar=ave_in_support[p],
                                   mu_0=ave_in_baseline,
                                   S=std_in_support[p],
                                   n=n_in_support[p])
        new_row['p_val_%s' % this_type] = '%.2e' % p_val

    # update results DataFrame
    df_res = df_res.append(new_row, ignore_index=True)

print(df_res)

# %%
