"""

"""

import numpy as np
import matplotlib.pyplot as plt

import mne

from dripp.trunc_norm_kernel.model import TruncNormKernel


def plot_cdl_atoms(dict_global, cdl_params, info,
                   n_top_atoms=None, plotted_atoms='all',
                   alpha_threshold=None,
                   plot_psd=False, plot_intensity=False,
                   df_res_dripp=None, plotted_tasks=None,
                   save_fig=True, path_fig=None):
    """

    Parameters
    ----------


    n_top_atoms : int | None
        number of atoms to plot when ordered by their ratio alpha / mu
        default is None

    plotted_atoms : list of int | 'all'
        list of indices of atoms to plot
        if 'all', plot all learned atoms
        default is 'all'

    plot_psd : bool
        if True, plot the Power Spectral Density (PSD) of the atom
        default if False

    plot_intensity : bool
        if True, plot the learned DriPP intensity function
        if True, df_res_dripp and plotted_tasks must be not None
        default is False

    df_res_dripp : pandas.DataFrame

    plotted_tasks : dict

    """

    #
    fontsize = 8
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "xtick.labelsize": fontsize,
        'ytick.labelsize': fontsize,
    })

    if plotted_atoms == 'all':
        plotted_atoms = range(cdl_params['n_atoms'])

    # number of plot per atom
    n_plots = 2 + plot_psd + plot_intensity
    # number of atoms to plot
    if n_top_atoms is not None:
        n_atoms = n_top_atoms
        # compute ratio (alpha / baseline)
        df_temp = df_res_dripp.copy()
        df_temp['ratio'] = df_temp['alpha_hat'].apply(
            np.array) / df_temp['baseline_hat']
        df_temp['ratio_max'] = df_temp['ratio'].apply(max)
        # keep only where the m_hat associated with the ratio max is positive
        df_temp['argmax_ratio'] = df_temp['ratio'].apply(np.argmax)
        df_temp = df_temp[df_temp.apply(
            lambda x: x.m_hat[x.argmax_ratio] > 0, axis=1)]
        # keep only where the m_hat associated with the ratio max is in support
        upper = df_res_dripp['upper'][0]
        df_temp = df_temp[df_temp.apply(
            lambda x: x.m_hat[x.argmax_ratio] < upper, axis=1)]
        # get top atoms
        df_temp = df_temp.sort_values(
            by=['ratio_max'], ascending=False)[:n_top_atoms]
        if alpha_threshold is not None:
            # hard filter on alpha estimate
            df_temp = df_temp[df_temp['alpha_hat'].apply(
                max) > alpha_threshold]
        # only plot those atoms
        plotted_atoms = df_temp['atom'].values
        n_top_atoms = min(n_top_atoms, len(plotted_atoms))
    else:
        n_atoms = len(plotted_atoms)
    # shape of the final figure
    n_columns = min(6, n_atoms)
    split = int(np.ceil(n_atoms / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
    # get CDL results
    u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])  # spatial
    v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])  # temporal
    sfreq = cdl_params['sfreq']
    atom_duration = v_hat_.shape[-1] / sfreq

    # x axis for temporal pattern
    n_times_atom = cdl_params.get('n_times_atom', int(round(sfreq * 1.0)))
    t = np.arange(n_times_atom) / sfreq

    if plot_intensity:
        # x axis for estimated intensity function
        lower = df_res_dripp['lower'][0]
        upper = df_res_dripp['upper'][0]
        xx = np.linspace(0, upper, 800)

    for ii, kk in enumerate(plotted_atoms):

        # Select the axes to display the current atom
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        # Select the current atom
        u_k = u_hat_[kk]
        v_k = v_hat_[kk]

        # next atom to plot
        ax = next(it_axes)
        ax.set_title('Atom % d' % kk, fontsize=fontsize, pad=0)

        # Plot the spatial map of the atom using mne topomap
        mne.viz.plot_topomap(data=u_k, pos=info, axes=ax, show=False)
        if i_col == 0:
            ax.set_ylabel('Spatial', labelpad=58, fontsize=fontsize)

        # Plot the temporal pattern of the atom
        ax = next(it_axes)
        ax.plot(t, v_k)
        ax.set_xlim(0, atom_duration)
        if i_col == 0:
            ax.set_ylabel('Temporal', fontsize=fontsize)

        # Power Spectral Density (PSD)
        if plot_psd:
            ax = next(it_axes)
            psd = np.abs(np.fft.rfft(v_k, n=256)) ** 2
            frequencies = np.linspace(0, sfreq / 2.0, len(psd))
            ax.semilogy(frequencies, psd, label="PSD", color="k")
            ax.set_xlim(0, 40)  # crop x axis
            ax.set_xlabel("Frequencies (Hz)", fontsize=fontsize)
            ax.grid(True)
            if i_col == 0:
                ax.set_ylabel("Power Spectral Density", labelpad=13,
                              fontsize=fontsize)

        # plot the estimate intensity function
        if plot_intensity:
            # select sub-df of interest
            df_atom = df_res_dripp[(df_res_dripp['atom'] == kk)]
            # df_temp = df_res[(df_res['atom'] == kk)
            #                  & (df_res['lower'] == lower)
            #                  & (df_res['upper'] == upper)
            #                  & (df_res['threshold'] == threshold)
            #                  & (df_res['shift_acti'] == shift_acti)]
            # # in case that there has been an early stopping
            # n_iter_temp = min(n_iter, df_temp['n_iter'].values.max())
            # df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]

            ax = next(it_axes)
            for jj, label in enumerate(plotted_tasks.keys()):
                # unpack parameters estimates
                alpha = list(df_atom['alpha_hat'])[0][jj]
                baseline = list(df_atom['baseline_hat'])[0]
                m = list(df_atom['m_hat'])[0][jj]
                sigma = list(df_atom['sigma_hat'])[0][jj]

                # define kernel function
                kernel = TruncNormKernel(lower, upper, m, sigma)
                yy = baseline + alpha * kernel.eval(xx)
                # lambda_max = baseline + alpha * kernel.max
                # ratio_lambda_max = lambda_max / baseline

                if i_col == 0:
                    ax.plot(xx, yy, label=label)
                else:
                    ax.plot(xx, yy)

            ax.set_xlim(0, upper)
            ax.set_xlabel('Time (s)', fontsize=fontsize)

            # share y scale
            if ii == 0:
                intensity_ax = ax
            else:
                intensity_ax.get_shared_y_axes().join(intensity_ax, ax)
                ax.autoscale()

            if i_col == 0:
                ax.set_ylabel('Intensity', labelpad=15, fontsize=fontsize)
                ax.legend(fontsize=fontsize, handlelength=1)

    fig.tight_layout()
    # save figure
    if save_fig:
        plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    return fig
