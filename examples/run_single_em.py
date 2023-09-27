from dripp.cdl.utils import get_dict_global
from dripp.dripp_model import DriPP

dataset = "somato"
dict_global = get_dict_global(dataset=dataset)

id_atom = 0  # run DriPP on first atom
events_tt = dict_global["events_tt"]
activations_tt = dict_global["activations_tt"][id_atom]

dripp_model = DriPP(
    lower=0,
    upper=2,
    sfreq=dict_global["sfreq"],
    use_dis=True,
    initializer="random",
    alpha_pos=True,
    n_iter=20,
    verbose=True,
    disable_tqdm=False,
    compute_loss=True,
)

dripp_model.fit(acti_tt=activations_tt, driver_tt=events_tt, T=dict_global["T"])
