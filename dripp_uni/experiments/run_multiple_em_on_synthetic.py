"""
Generates data and runs the EM algorithm for several parameter combinations.
"""

import numpy as np
import pandas as pd
import itertools
import time
from tqdm import tqdm
from joblib import Memory, Parallel, delayed

from dripp_uni.config import CACHEDIR, SAVE_RESULTS_PATH
from dripp_uni.trunc_norm_kernel.simu import simulate_data
from dripp_uni.trunc_norm_kernel.optim import em_truncated_norm
from dripp_uni.trunc_norm_kernel.model import TruncNormKernel

memory = Memory(CACHEDIR, verbose=0)


def procedure(comb_simu, combs_em, T_max, simu_params, simu_params_to_vary,
              em_params, em_params_to_vary, sfreq=150.):
    """For a given set of simulation parameters, simulate the data and run the
    EM for several combination of EM parameters

    Parameters
    ----------
    comb_simu : tuple
        a combination of simulation parameters values

    combs_em : list of tuple
        a list of all possible combinations of em parameters values

    T_max : int | float
        maximum length for data simulation

    simu_params : dict
        dictionary of all simulation parameters and their values

    simu_params_to_vary : dict
        dictionary of all simulation parameters we want to vary (keys) and
        their corresponding set of values

    em_params : dict
        dictionary of all EM parameters and their values

    em_params_to_vary : dict
        dictionary of all EM parameters we want to vary (keys) and
        their corresponding set of values

    sfreq : int | None
        sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    Returns
    -------
    new_rows : list of dict

    """
    # update data simulation parameters
    simu_params_temp = simu_params.copy()
    for i, param in enumerate(simu_params_to_vary.keys()):
        simu_params_temp[param] = comb_simu[i]
    # simulate data of duration T_max
    driver_tt_, acti_tt_, _ = simulate_data(T=T_max, sfreq=sfreq,
                                            **simu_params_temp)
    # define true kernel
    kernel_simu = TruncNormKernel(simu_params_temp['lower'],
                                  simu_params_temp['upper'],
                                  simu_params_temp['m'],
                                  simu_params_temp['sigma'],
                                  sfreq)
    # for every combination of EM parameters
    new_rows = []
    for this_comb_em in combs_em:
        # update EM algorithm parameters
        em_params_temp = em_params.copy()
        for i, param in enumerate(em_params_to_vary.keys()):
            em_params_temp[param] = this_comb_em[i]

        # crop process for the given length of T
        T = em_params_temp['T']
        acti_tt = acti_tt_[acti_tt_ < T]
        # respect no-overlapping assumption
        driver_tt = driver_tt_[driver_tt_ < T - em_params_temp['upper']]

        # run EM
        start_time = time.time()
        res_params, _, hist_loss = em_truncated_norm(acti_tt, driver_tt,
                                                     disable_tqdm=True,
                                                     sfreq=sfreq,
                                                     **em_params_temp)
        comput_time = time.time() - start_time
        baseline_hat, alpha_hat, m_hat, sigma_hat = res_params

        # define estimated kernel
        kernel_hat = TruncNormKernel(
            em_params_temp['lower'], em_params_temp['upper'],
            m_hat, sigma_hat, sfreq)

        # compute infinite norm between true en estimated intensity functions
        lower_ = min(simu_params_temp['lower'], em_params_temp['lower'])
        upper_ = max(simu_params_temp['upper'], em_params_temp['upper'])

        xx = np.linspace(lower_ - 1, upper_ + 1, 500)
        yy_true = simu_params_temp['baseline'] + \
            simu_params_temp['alpha'] * kernel_simu.eval(xx)
        yy_hat = baseline_hat + alpha_hat * kernel_hat.eval(xx)
        inf_norm = abs(yy_true - yy_hat).max()

        lambda_max = simu_params_temp['baseline'] + \
            simu_params_temp['alpha'] * kernel_simu.max
        inf_norm_rel = inf_norm / lambda_max

        # add new row
        new_row = simu_params_temp.copy()
        if new_row['n_tasks'] is None:
            new_row['n_tasks'] = 1

        new_row['lower_em'] = em_params_temp['lower']
        new_row['upper_em'] = em_params_temp['upper']
        new_row['T'] = em_params_temp['T']
        new_row['baseline_hat'] = baseline_hat
        new_row['alpha_hat'] = alpha_hat
        new_row['m_hat'] = m_hat
        new_row['sigma_hat'] = sigma_hat
        new_row['n_driver_tt'] = len(driver_tt)
        new_row['sfreq'] = sfreq
        new_row['infinite_norm_of_diff'] = inf_norm
        new_row['infinite_norm_of_diff_rel'] = inf_norm_rel
        new_row['comput_time'] = comput_time
        new_row['n_iter_real'] = len(hist_loss)

        new_rows.append(new_row)

    return new_rows


@memory.cache(ignore=['n_jobs'])
def run_multiple_em_on_synthetic(simu_params, simu_params_to_vary,
                                 em_params, em_params_to_vary,
                                 sfreq=150., n_jobs=6, save_results=False):
    """Run several EM in parallel for multiple simulation parameters
    combinations and for multiple EM parameters combinations

    Parameters
    ----------
    simu_params : dict
        dictionary of all simulation parameters and their values

    simu_params_to_vary : dict
        dictionary of all simulation parameters we want to vary (keys) and
        their corresponding set of values

    em_params : dict
        dictionary of all EM parameters and their values

    em_params_to_vary : dict
        dictionary of all EM parameters we want to vary (keys) and
        their corresponding set of values

    sfreq : int | None
        sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    n_jobs : int
        The maximum number of concurrently running jobs
        default is 6

    Returns
    -------
    pandas.DataFrame

    """
    # get all parameters combination for data simulation
    combs_simu = list(itertools.product(*list(simu_params_to_vary.values())))
    # get all parameters combination for EM algorithm
    combs_em = list(itertools.product(*list(em_params_to_vary.values())))
    # get maximum duration, to simulate with
    T_max = em_params_to_vary['T'].max()

    # define dataframe to stock results
    df_res = pd.DataFrame()

    # run in parallel
    if n_jobs > 1:
        new_rows = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(procedure)(this_comb_simu, combs_em, T_max,
                               simu_params, simu_params_to_vary,
                               em_params, em_params_to_vary, sfreq)
            for this_comb_simu in combs_simu)
    # run linearly
    else:
        new_rows = []
        for this_comb_simu in tqdm(combs_simu):
            new_row = procedure(this_comb_simu, combs_em, T_max,
                                simu_params, simu_params_to_vary,
                                em_params, em_params_to_vary, sfreq)
            new_rows.append(new_row)

    for this_new_row in new_rows:
        df_res = df_res.append(this_new_row, ignore_index=True)

    if save_results:
        # save dataframe as csv
        path_df_res = SAVE_RESULTS_PATH
        if not path_df_res.exists():
            path_df_res.mkdir(parents=True)

        df_res.to_csv(path_df_res / 'results_em_synthetic.csv')

    return df_res
