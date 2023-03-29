import numpy as np
import pandas as pd
from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids

DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

subject_id = "CC620264"  # 76.33 Female
atom_duration = 0.5
sfreq = 150.

print("Loading the data...", end=' ', flush=True)
bp = BIDSPath(
    root=BIDS_ROOT,
    subject=subject_id,
    task="smt",
    datatype="meg",
    extension=".fif",
    session="smt",
)
raw = read_raw_bids(bp)
print('done')
# get age and sex of the subject
participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
age, sex = participants[participants['participant_id']
                        == 'sub-' + str(subject_id)][['age', 'sex']].iloc[0]
print(f'Subject {subject_id}: {age} years-old {sex}')


print("Preprocessing the data...", end=' ', flush=True)
raw.load_data()
raw.filter(l_freq=None, h_freq=125)
raw.notch_filter([50, 100])
raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                       cross_talk=CT_SPARSE_FILE,
                                       st_duration=10.0)
raw.pick(['grad', 'stim'])
events, event_id = mne.events_from_annotations(raw)
# event_id = {'audiovis/1200Hz': 1,
#             'audiovis/300Hz': 2,
#             'audiovis/600Hz': 3,
#             'button': 4,
#             'catch/0': 5,
#             'catch/1': 6}
raw.filter(l_freq=2, h_freq=45)

raw, events = raw.resample(
    sfreq, npad='auto', verbose=False, events=events)
# Set the first sample to 0 in event stim
events[:, 0] -= raw.first_samp
print('done')

X = raw.get_data(picks=['meg'])

cdl_params = {
    'subject_id': subject_id,
    'use_greedy': True,
    'n_atoms': 20,
    'n_times_atom': int(np.round(atom_duration * sfreq)),
    'sfreq': sfreq,
    'n_iter': 100,
    'eps': 1e-5,
    'reg': 0.2,
    'tol_z': 1e-3,
    'n_jobs': 5,
    'n_splits': 10
}
