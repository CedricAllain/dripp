"""
Functions to plot results obtained with CDL
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import mne

from .utils import apply_threshold


def plot_atoms(u, v, info, plotted_atoms='all', sfreq=150., fig_name=None):
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
    n_plots = 2
    n_columns = min(6, len(plotted_atoms))
    split = int(np.ceil(len(plotted_atoms) / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

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

    fig.tight_layout()

    if fig_name:  # save figure
        path_fig = fig_name
        plt.savefig(path_fig + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(path_fig + ".png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


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
