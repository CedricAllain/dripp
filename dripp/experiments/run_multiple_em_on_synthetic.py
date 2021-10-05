"""
Generates data and runs the EM algorithm for several parameter combinations.
"""

import numpy as np
import pandas as pd
import itertools
import time
from tqdm import tqdm
from joblib import Memory, Parallel, delayed

from dripp.config import CACHEDIR, SAVE_RESULTS_PATH
from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.optim import em_truncated_norm
# from dripp.trunc_norm_kernel.model import Intensity, TruncNormKernel
from dripp.trunc_norm_kernel.model import TruncNormKernel
from dripp.trunc_norm_kernel.utils import convert_variable_multi

memory = Memory(CACHEDIR, verbose=0)


def procedure(comb_simu, combs_em, T_max, simu_params, simu_params_to_vary,
              em_params, em_params_to_vary, sfreq=150., n_drivers=1):
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
    simu_data = simulate_data(
        T=T_max, sfreq=sfreq, n_drivers=n_drivers, return_nll=False,
        **simu_params_temp)
    driver_tt_, acti_tt_, kernel_simu = simu_data[0:3]
    # define true kernel(s) used for data simulation
    # kernel_simu = []
    # lower_simu = convert_variable_multi(
    #     simu_params_temp['lower'], n_drivers, repeat=True)
    # upper_simu = convert_variable_multi(
    #     simu_params_temp['upper'], n_drivers, repeat=True)
    # m_simu = convert_variable_multi(
    #     simu_params_temp['m'], n_drivers, repeat=True)
    # sigma_simu = convert_variable_multi(
    #     simu_params_temp['sigma'], n_drivers, repeat=True)
    # for p in range(n_drivers):
    #     kernel_simu.append(TruncNormKernel(
    #         lower_simu[p], upper_simu[p], m_simu[p], sigma_simu[p],
    #         sfreq=sfreq))
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
        # mask = driver_tt_ < T
        # list_tt = []
        # n_tt = []
        # for i, this_mask in enumerate(mask):
        #     this_driver_tt = list(driver_tt_[i][this_mask])
        #     n_tt.append(len(this_driver_tt))
        #     list_tt.append(this_driver_tt)
        # driver_tt = np.array([tt[:min(n_tt)] for tt in list_tt])
        # respect no-overlapping assumption
        # driver_tt = driver_tt_[driver_tt_ < T - em_params_temp['upper']]
        driver_tt = [tt[tt < T] for tt in driver_tt_]

        # run EM
        start_time = time.time()
        res_params, history_params, _ = em_truncated_norm(
            acti_tt, driver_tt, disable_tqdm=True, sfreq=sfreq,
            **em_params_temp)
        comput_time = time.time() - start_time
        baseline_hat, alpha_hat, m_hat, sigma_hat = res_params

        # define estimated kernel

        # kernel_hat = TruncNormKernel(
        #     em_params_temp['lower'], em_params_temp['upper'],
        #     m_hat, sigma_hat, sfreq)
        kernel_hat = []
        lower_em = convert_variable_multi(
            em_params_temp['lower'], n_drivers, repeat=True)
        upper_em = convert_variable_multi(
            em_params_temp['upper'], n_drivers, repeat=True)
        for p in range(n_drivers):
            kernel_hat.append(TruncNormKernel(
                lower_em[p], upper_em[p], m_hat[p], sigma_hat[p], sfreq=sfreq))

        # create new row
        new_row = simu_params_temp.copy()
        if new_row['n_tasks'] is None:
            new_row['n_tasks'] = 1  # 100% of kept events

        new_row['lower_em'] = em_params_temp['lower']
        new_row['upper_em'] = em_params_temp['upper']
        new_row['T'] = em_params_temp['T']
        new_row['baseline_hat'] = baseline_hat
        new_row['alpha_hat'] = alpha_hat
        new_row['m_hat'] = m_hat
        new_row['sigma_hat'] = sigma_hat
        new_row['sfreq'] = sfreq
        new_row['comput_time'] = comput_time
        # true number of iterations
        new_row['n_iter_real'] = len(history_params[0])
        # percentage of common timestamps
        if n_drivers > 1:
            common_tt = set(driver_tt[0])
            for p in range(1, n_drivers):
                common_tt = common_tt.intersection(driver_tt[1])
            # ppt_common = len(common_tt) / driver_tt.shape[1]
            new_row['n_common'] = len(common_tt)
            # new_row['ppt_common'] = ppt_common

        # for each kernel, compute the relative infinite norm
        for p in range(n_drivers):
            lower_ = min(kernel_simu[p].lower, lower_em[p])
            upper_ = max(kernel_simu[p].upper, upper_em[p])

            xx = np.linspace(lower_ - 1, upper_ + 1, 500)
            yy_true = simu_params_temp['baseline'] + \
                simu_params_temp['alpha'][p] * kernel_simu[p].eval(xx)
            yy_hat = baseline_hat + alpha_hat[p] * kernel_hat[p].eval(xx)
            inf_norm = abs(yy_true - yy_hat).max()
            lambda_max = simu_params_temp['baseline'] + \
                simu_params_temp['alpha'][p] * kernel_simu[p].max
            inf_norm_rel = inf_norm / lambda_max
            new_row['infinite_norm_of_diff_kernel_%i' % p] = inf_norm
            new_row['infinite_norm_of_diff_rel_kernel_%i' %
                    p] = inf_norm_rel
            # add number of events created for each driver
            new_row['n_driver_tt_kernel_%i' % p] = len(driver_tt[p])

        # add new row
        new_rows.append(new_row)

        # compute infinite norm between true en estimated intensity functions
        # if only on driver, the infinite norm is reduced at a kernel support
        # if n_drivers == 1:
        #     lower_ = min(*lower_simu, *lower_em)
        #     upper_ = max(*upper_simu, *upper_em)

        #     xx = np.linspace(lower_ - 1, upper_ + 1, 500)
        #     yy_true = simu_params_temp['baseline'] + \
        #         simu_params_temp['alpha'] * kernel_simu[0].eval(xx)
        #     yy_hat = baseline_hat + alpha_hat * kernel_hat[0].eval(xx)
        #     inf_norm = abs(yy_true - yy_hat).max()

        #     lambda_max = simu_params_temp['baseline'] + \
        #         simu_params_temp['alpha'][0] * kernel_simu[0].max

        #     # relative infinite norm (for easy EM comparisons)
        #     inf_norm_rel = inf_norm / lambda_max
        #     # add metric to new row
        #     new_row['infinite_norm_of_diff'] = inf_norm
        #     new_row['infinite_norm_of_diff_rel'] = inf_norm_rel
        # else:  # n_drivers >= 2
        # xx = np.linspace(0, T, T * 500)
        # intensity_true = Intensity(
        #     simu_params_temp['baseline'], simu_params_temp['alpha'],
        #     kernel_simu, driver_tt_)
        # yy_true = intensity_true(xx)
        # intensity_hat = Intensity(
        #     baseline_hat, alpha_hat, kernel_hat, driver_tt_)
        # yy_hat = intensity_hat(xx)
        # inf_norm = abs(yy_true - yy_hat).max()
        # lambda_max = yy_true.max()

    return new_rows


@memory.cache(ignore=['n_jobs'])
def run_multiple_em_on_synthetic(simu_params, simu_params_to_vary,
                                 em_params, em_params_to_vary,
                                 sfreq=150., n_drivers=1, n_jobs=6,
                                 save_results=False):
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
                               em_params, em_params_to_vary, sfreq, n_drivers)
            for this_comb_simu in combs_simu)
    # run linearly
    else:
        new_rows = []
        for this_comb_simu in tqdm(combs_simu):
            new_row = procedure(this_comb_simu, combs_em, T_max,
                                simu_params, simu_params_to_vary,
                                em_params, em_params_to_vary, sfreq, n_drivers)
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
