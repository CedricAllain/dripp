# %%
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids

DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)

# %%


def get_age_cat(age):
    age_bins = [18]
    age_bins.extend([28 + 10*i for i in range(7)])
    age_bins.append(127)

    return np.searchsorted(age_bins, age, side='right')


participants['id_cat'] = participants['participant_id'].str[6].astype(int)
participants['age_cat'] = participants['age'].apply(lambda x: get_age_cat(x))

participants['same_cat'] = (participants['id_cat'] == participants['age_cat'])
print(f'number of misclassifications: {len(participants[~participants['same_cat']])}')
# %%
print('Column age_cat')
for cat in range(1, 9):
    sub_df = participants[participants['age_cat'] == cat]
    age_min = sub_df['age'].min()
    age_max = sub_df['age'].max()
    print(f"cat {cat}: min = {age_min}, max = {age_max} ({len(sub_df)} subjects)")

print('Column id_cat')
for cat in range(1, 9):
    sub_df = participants[participants['id_cat'] == cat]
    age_min = sub_df['age'].min()
    age_max = sub_df['age'].max()
    print(f"cat {cat}: min = {age_min}, max = {age_max} ({len(sub_df)} subjects)")

# %%
