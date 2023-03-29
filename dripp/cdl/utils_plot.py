"""
Functions to plot results obtained with CDL
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import mne

from dripp.cdl.utils import apply_threshold
from dripp.trunc_norm_kernel.model import TruncNormKernel


def plot_atoms(u, v, info, plotted_atoms='all', sfreq=150., fig_name=None, df_res=None):
    """Plot spatial and temporal representations of learned atoms.

    Parameters
    ----------
    u, v : array-like XXX change for cdl object (instance or dict)

    info : dict
        MNE dictionary of information about recording settings.

    plotted_atoms : int, list, 'all'
        if int, the number of atoms to plots (e.g., if set to 5, plot the first
        5 atoms)
        if list, the list of atom indexes to plot
        if 'all', plot all the learned atom
        defaults is 'all'

    sfreq : float
        sampling frequency, the signal will be resampled to match this.

    fig_name : str | None

    df_res : pandas DataFrame with DriPP results

    Returns
    -------
    None

    """

    n_atoms, n_times_atom = u.shape[0], int(v.shape[1])

    if plotted_atoms == 'all':
        plotted_atoms = range(n_atoms)
    elif isinstance(plotted_atoms, int):
        plotted_atoms = range(plotted_atoms)

    t = np.arange(n_times_atom) / sfreq

    # number of plots by atom
    if df_res is not None:
        plot_intensity = True
        df_ratio = []
    else:
        plot_intensity = False

    n_plots = 2 + plot_intensity
    n_columns = min(6, len(plotted_atoms))
    split = int(np.ceil(len(plotted_atoms) / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
    if n_columns == 1:
        axes = np.atleast_2d(axes).T

    for ii, kk in enumerate(plotted_atoms):

        # Select the axes to display the current atom
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        # Select the current atom
        v_k = v[kk, :]
        u_k = u[kk, :]

        # Plot the spatial map of the atom using mne topomap
        ax = next(it_axes)
        ax.set_title('Atom % d' % kk, pad=0)

        mne.viz.plot_topomap(data=u_k, pos=info, axes=ax, show=False)
        if i_col == 0:
            ax.set_ylabel('Spatial', labelpad=28)

        # Plot the temporal pattern of the atom
        ax = next(it_axes)
        ax.plot(t, v_k)
        ax.set_xlim(min(t), max(t))
        if i_col == 0:
            ax.set_ylabel('Temporal')

        # Plot the learned intensities
        if not plot_intensity:
            continue
        # else, plot the intensities
        ax = next(it_axes)

        plotted_tasks = df_res['tasks'][0]
        df_temp = df_res[(df_res['atom'] == kk)]
        lower, upper = df_temp['lower'].iloc[0], df_temp['upper'].iloc[0]
        xx = np.linspace(0, upper, int((upper-lower)*1_000))
        baseline = list(df_temp['baseline_hat'])[0]
        this_atom_ratio = 0
        this_atom_label = 'N/A'
        for jj, label in enumerate(plotted_tasks):
            # unpack parameters estimates
            alpha = list(df_temp['alpha_hat'])[0][jj]
            m = list(df_temp['m_hat'])[0][jj]
            sigma = list(df_temp['sigma_hat'])[0][jj]
            # compute ratio
            temp_ratio = alpha / baseline
            if temp_ratio > this_atom_ratio:
                this_atom_ratio = temp_ratio
                this_atom_label = label
            # define kernel function
            kernel = TruncNormKernel(lower, upper, m, sigma)
            yy = baseline + alpha * kernel.eval(xx)
            if ii > 0:
                plot_label = None
            else:
                plot_label = label
            # plot intensity
            ax.plot(xx, yy, label=plot_label)

        ax.set_xlim(min(xx), max(xx))
        if i_col == 0:
            ax.set_ylabel('Intensity')
        ax.set_xlabel('Time (s)')
        if plot_label:
            ax.legend()

        df_ratio.append({'atom': kk, 'ratio': this_atom_ratio,
                        'label': this_atom_label})

    if plot_intensity:
        df_ratio = pd.DataFrame(data=df_ratio).sort_values(by='ratio')
        top_atoms = df_ratio['atom'][:5].values
        print(f"Top 5 atoms: {top_atoms}")

    fig.tight_layout()

    if fig_name:  # save figure
        path_fig = fig_name
        plt.savefig(path_fig + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(path_fig + ".png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    if plot_intensity:
        return df_ratio


def plot_z_boxplot(z_hat, p_threshold=0, per_atom=True,
                   yscale='log', add_points=True, add_number=True,
                   fig_name=None):
    """
    Plot activations boxplot for each atom, with a possible thresholding.
    """
    n_atoms = z_hat.shape[1]
    values = apply_threshold(
        z=z_hat, p_threshold=p_threshold, per_atom=per_atom)

    df_z = pd.DataFrame(data=values).T
    df_z = df_z.rename(columns={k: f'Values{k}' for k in range(n_atoms)})
    df_z["id"] = df_z.index
    df_z = pd.wide_to_long(df_z, stubnames=['Values'], i='id', j='Atom')\
             .reset_index()[['Atom', 'Values']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale(yscale)

    sns.boxplot(x='Atom', y='Values', data=df_z)

    if add_points:
        sns.stripplot(
            x="Atom", y="Values", data=df_z, size=2, color=".3", linewidth=0)

    if add_number:
        ax2 = ax.twinx()
        xx = list(range(n_atoms))
        yy = [len(z) for z in values]
        ax2.plot(xx, yy, color="black", alpha=.8)
        ax2.set_ylim(0)
        ax2.set_ylabel('# non-nul activations')

    plt.xticks(rotation=45)
    title = "Activations repartition"
    if p_threshold > 0:
        title += f" with {'per-atom' if per_atom else 'global'} thresholding of {p_threshold}%"
    plt.title(title)

    if fig_name:
        plt.savefig(fig_name)

    plt.show()


def plot_z_threshold_effect(z_hat, list_threshold=[0, 25, 50, 75],
                            per_atom=True, save=False,
                            fig_name='z_threshold_effect.png'):
    """
    Plot the number of non-null activations for each atom for multiple levels
    of thresholding.
    """
    n_atoms = z_hat.shape[1]

    xx = list(range(n_atoms))
    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(
        0, 1, len(list_threshold)+1)][1:]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, p_threshold in enumerate(list_threshold):
        z_threshold = apply_threshold(z_hat, p_threshold, per_atom=per_atom)
        yy = [len(z) for z in z_threshold]
        ax.plot(xx, yy, color=palette[i], label=p_threshold)

    ax.set_xlim(0, n_atoms-1)
    ax.set_xticks(xx)
    ax.set_ylim(0)
    ax.legend(title='percentage')
    ax.set_xlabel('Atom')
    ax.set_ylabel('# non-nul activations')

    plt.xticks(rotation=45)
    plt.title(f"Effect of {'per-atom' if per_atom else 'global'} thresholding")

    if save:
        plt.savefig(fig_name)

    plt.show()


def plot_acti_tt_boxplot(acti_tt):
    """

    """

    n_atoms = len(acti_tt)

    df_acti = pd.DataFrame(data=acti_tt).T
    df_acti = df_acti.rename(columns={k: f'Time{k}' for k in range(n_atoms)})
    df_acti["id"] = df_acti.index
    df_acti = pd.wide_to_long(df_acti, stubnames=['Time'], i='id', j='Atom')\
        .reset_index()[['Atom', 'Time']]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(x='Atom', y='Time', data=df_acti, orient='h')
    plt.show()


def plot_events_distribution(dict_events=None, dict_atoms=None):
    """Plot, on the same figure, the distributions of a list of events and a
    list of atoms' activations.

    Parameters
    ----------

    dict_events : dict
        keys: string, label of the events
        values: array-like, events' timestamps

    dict_events : dict
        keys: string, label of the atoms' activation
        values: array-like, atoms' activation's timestamps

    Returns
    -------
    """
