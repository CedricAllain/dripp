# %%
from dripp.cdl.run_cdl import run_cdl
from utils import post_process_cdl
from utils_plot import plot_atoms, plot_z_boxplot

# %%

DATASET = "somato"  # 'sample' | 'somato' | 'camcan

# run CDL with default parameters
dict_res = run_cdl(DATASET)
u_hat, v_hat, z_hat = dict_res["u_hat_"], dict_res["v_hat_"], dict_res["z_hat"]
sfreq = dict_res["sfreq"]

# %%
# plot all extracted atoms
plot_atoms(
    u_hat,
    v_hat,
    info=dict_res["info"],
    plotted_atoms="all",
    sfreq=sfreq,
    fig_name=f"{DATASET}_all_atoms",
)

# plot activation distribution
plot_z_boxplot(
    z_hat,
    p_threshold=10,
    per_atom=True,
    yscale="log",
    add_points=False,
    add_number=True,
    fig_name=f"{DATASET}_z_boxplot.png",
)

# post-processing
post_process_params = dict(
    time_interval=0.01, threshold=10, percent=True, per_atom=True
)


events_tt, atoms_tt = post_process_cdl(
    events=dict_res["events"],
    event_id=dict_res["event_id"].values(),
    v_hat_=v_hat,
    z_hat=z_hat,
    sfreq=sfreq,
    post_process_params=post_process_params,
)

# %%
