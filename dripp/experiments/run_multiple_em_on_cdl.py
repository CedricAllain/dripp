import itertools
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from dripp.cdl import utils
from dripp.experiments.run_cdl import run_cdl_sample, run_cdl_somato,\
    run_cdl_camcan
from dripp.trunc_norm_kernel.optim import em_truncated_norm
from dripp.config import CACHEDIR, SAVE_RESULTS_PATH

memory = Memory(CACHEDIR, verbose=0)


def procedure(comb):
    """Procedure to parallelized

    Parameters
    ----------
    comb : tuple
        tuple ((atom, task), args) on which to perform the EM algorithm
        where,
            atom : int, the atom idex
            task : int | array-like, ids of tasks
            args : dict, dictionary of EM parameters, with following keys
                lower, upper : int | float
                T : int | float
                initializer : str
                early_stopping : str | None
                early_stopping_params : dict | None
                alpha_pos : bool
                n_iter : int | array-like
                    if array-like, returns the value of learned parameters at
                    the different values of n_iter

    Return
    ------
    new_row : dict | list of dict
        new row(s) of the results DataFrame
        return a list of dict if n_iter's type is array-like

    """
    (atom, tasks), args = comb

    n_bootstrap = args['n_bootstrap']
    p_bootstrap = args['p_bootstrap']

    # n_iter = np.atleast_1d(args['n_iter'])
    # n_iter_max = max(n_iter)
    n_iter = args['n_iter']

    # get activation timestamps
    atoms_timestamps = np.array(args['atoms_timestamps'])
    aa = atoms_timestamps[atom]

    # get and merge tasks timestamps
    events_timestamps = args['events_timestamps']  # dict

    def proprocess_tasks(tasks):
        """
        XXX
        """
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

    # base row
    base_row = {'atom': int(atom),
                'tasks': tasks,
                'lower': args['lower'],
                'upper': args['upper'],
                'initializer': args['initializer'],
                'n_bootstrap': n_bootstrap,
                'p_bootstrap': p_bootstrap}

    # create new DataFrame rows
    new_rows = []

    for _ in range(n_bootstrap):

        if p_bootstrap < 1:
            size = int(len(tt) * p_bootstrap)
            tt_ = np.random.choice(tt, size, replace=False)
            tt_ = np.sort(tt_)
        else:
            tt_ = tt.copy()

        # run EM algorithm
        res_em = em_truncated_norm(
            acti_tt=aa,
            driver_tt=tt_,
            lower=args['lower'],
            upper=args['upper'],
            T=args['T'],
            initializer=args['initializer'],
            early_stopping=args['early_stopping'],
            early_stopping_params=args['early_stopping_params'],
            alpha_pos=args['alpha_pos'],
            # n_iter=n_iter_max,
            n_iter=n_iter,
            verbose=True,
            disable_tqdm=True)

        # get results
        res_params, history_params, history_loss = res_em

        # unpack parameters history
        # hist_baseline, hist_alpha, hist_m, hist_sigma = history_params
        # list of values for n_iter that exist
        # list_n_iter = [n for n in n_iter if n < hist_baseline.size]
        # list_n_iter = n_iter[n_iter < hist_baseline.size]
        # make sure the last iteration will be added
        # if not (hist_baseline.size - 1) in list_n_iter:
        #     list_n_iter = np.append(list_n_iter, hist_baseline.size - 1)

        # for n in list_n_iter:
        #     new_row = {**base_row,
        #                'n_iter': n,
        #                'baseline_hat': hist_baseline[n],
        #                'alpha_hat': hist_alpha[n],
        #                'm_hat': hist_m[n],
        #                'sigma_hat': hist_sigma[n]}
        #     if len(history_loss) > 0:
        #         new_row['nll'] = history_loss[n]
        #     new_rows.append(new_row)

        baseline_hat, alpha_hat, m_hat, sigma_hat = res_params
        new_row = {**base_row,
                   'n_iter': n_iter,
                   'baseline_hat': baseline_hat,
                   'alpha_hat': alpha_hat,
                   'm_hat': m_hat,
                   'sigma_hat': sigma_hat,
                   'baseline_init': history_params[0][0],
                   'alpha_init': history_params[1][0]}

    # return new_rows
    return [new_row]


# @memory.cache(ignore=['n_jobs'])
def run_multiple_em_on_cdl(data_source='sample', cdl_params={},
                           shift_acti=True,
                           atom_to_filter=None, time_interval=0.01,
                           threshold=0.6e-10,
                           list_atoms=None, list_tasks=None,
                           n_driver=1,
                           lower=30e-3, upper=500e-3,
                           n_iter=400, initializer='smart_start',
                           early_stopping=None, early_stopping_params={},
                           alpha_pos=True, n_jobs=6,
                           n_bootstrap=1, p_bootstrap=1, save_results=False):
    """Run in parallel EM algorithm on results obtained from
    `dripp.experiments.run_cdl`, with several combination of (atoms, tasks).
    Results are returned on a pd.DataFrame object

    Parameters
    ----------
    cdl_params : dict
        Dictionary of keywords arguments for run_cdl() method

    shift_acti : boolean
        if True, shift activation timestamps by the atom's argmax

    atom_to_filter : list | None
        list of atom's indexes for which their activation will be filter
        Default is None

    time_interval : float
        Minimal time interval, in second, between two activations.
        Used if `atom_to_filter` is not None
        Default is 0.01

    threshold : float | array-like
        Threshold value(s) used to filter out insignificant activations.
        Default is 0.6e-10

    list_atoms : list
        list of atoms' indexes for which the EM algo will be computed.
        If None, then EM algo is computed for all atoms.
        Default is None

    lower, upper : float | array-like
        Truncation values for the kernel
        If array-like, EM will be computed with every couple (lower, upper)
        Default is 30e-3, 500e-3

    n_iter : int | array-like
        Number of iteration for the EM-based algorithm
        If array-like, denotes the different iteration values we are interested
        in. The EM will be computed with the maximum of that list.
        Default is 400

    initializer : str ('smart_start' | 'random')
        Initialization method
        Default is 'smart_start'

    early_stopping : string
        "early_stopping_sigma" | "early_stopping_percent_mass" | None
        method used for early stopping

    early_stopping_params : dict
        parameters for the early stopping method
        for 'early_stopping_sigma', keys must be 'n_sigma', 'n_tt' and 'sfreq'
        for 'early_stopping_percent_mass', keys must be 'alpha', 'n_tt' and
        'sfreq'

    alpha_pos : boolean
        if True, force alpha to be non-negative

    list_tasks : list
        list of task's indexes, or tasks (regrouped in a list), for wich the EM
        algo will be computed.
        If None, a value will be attributed based on the used dataset:
            for sample, [1, 2, 3, 4, [1, 2], [3, 4]]
            for camcan, [6, 7, 8, 9, [6, 7, 8]]
        Default is None

    n_jobs

    n_bootstrap : int

    p_bootstrap : float
        between 0 and 1, percentage of tasks events to keep

    Return
    ------
    dict_global : dict

    df_res : pandas.DataFrame

    """

    # get CDL results
    if data_source == 'sample':
        dict_global = run_cdl_sample(**cdl_params)
    elif data_source == 'somato':
        dict_global = run_cdl_somato(**cdl_params)
    elif data_source == 'camcan':
        dict_global = run_cdl_camcan(**cdl_params)

    # if not given, will run the EM for every atom extracted
    if list_atoms is None:
        n_atoms = dict_global['dict_cdl_params']['n_atoms']
        list_atoms = list(range(n_atoms))

    if list_tasks is None:
        if data_source == 'sample':
            list_tasks = [1, 2, 3, 4, [1, 2], [3, 4]]
        elif data_source == 'camcan':
            list_tasks = [6, 7, 8, 9, [6, 7, 8]]
        elif data_source == 'somato':
            list_tasks = [[1, 2, 3], 4]
        else:
            raise ValueError('list_tasks is None and data source is '
                             'unknown. '
                             'Please provide a list of tasks ids.')

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

    if atom_to_filter is not None:
        acti = utils.filter_activation(
            acti, atom_to_filter, sfreq, time_interval)

    df_res = pd.DataFrame()
    for t in np.atleast_1d(threshold):
        # get atoms' timestamps thresholded
        atoms_timestamps = utils.get_atoms_timestamps(acti=acti,
                                                      sfreq=sfreq,
                                                      threshold=t)

        procedure_kwargs = [{'events_timestamps': events_timestamps,
                             'atoms_timestamps': atoms_timestamps,
                             'lower': i, 'upper': j,
                             'T': T, 'initializer': initializer,
                             'early_stopping': early_stopping,
                             'early_stopping_params': early_stopping_params,
                             'alpha_pos': alpha_pos,
                             'n_iter': n_iter,
                             'n_bootstrap': n_bootstrap,
                             'p_bootstrap': p_bootstrap}
                            for i in np.atleast_1d(lower)
                            for j in np.atleast_1d(upper)]

        if n_driver == 1:
            combs_atoms_tasks = list(itertools.product(list_atoms, list_tasks))
        else:
            # n_driver = len(list_tasks)
            combs_atoms_tasks = [(kk, list_tasks) for kk in list_atoms]

        combs = list(itertools.product(combs_atoms_tasks, procedure_kwargs))

        if n_jobs == 1:
            # run in linear
            df_temp = pd.DataFrame()
            for this_comb in combs:
                new_rows = procedure(this_comb)
                df_temp = df_temp.append(new_rows, ignore_index=True)
        else:
            # run in parallel
            df_temp = pd.DataFrame()
            new_rows = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(procedure)(this_comb) for this_comb in combs)

            for new_row in new_rows:
                df_temp = df_temp.append(new_row, ignore_index=True)

        df_temp['threshold'] = t
        df_temp['shift_acti'] = shift_acti
        # concatenate DataFrames
        df_res = pd.concat([df_res, df_temp], ignore_index=True)

    if save_results:
        # save df_res as csv
        path_df_res = SAVE_RESULTS_PATH
        if not path_df_res.exists():
            path_df_res.mkdir(parents=True)

        df_res.to_csv(SAVE_RESULTS_PATH / 'results_em_sample_multi.csv')

    return dict_global, df_res
