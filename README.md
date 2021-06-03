# DriPP: Driven Point Processes to Model Stimuli Induced Patterns in M/EEG Signals

This repository is the official implementation of DriPP: Driven Point Processes to Model Stimuli Induced Patterns in M/EEG Signals.

## Install package under development 

To install the package under development, place yourself in the folder and run

```shell
pip install -e .
```

## Results

To get the results on synthetic data (figures 2, 3, A.1 and A.2 in the paper)

```shell
python eval_synthetic.py
```

> When run in parallel with 40 CPU, takes approximately

To get the results on [MNE sample dataset](https://mne.tools/dev/overview/datasets_index.html#sample) (figure 4 in paper)

```shell
python eval_sample.py
```

> When run in parallel with 40 CPU, takes approximately

To get the results on [MNE somatosensory dataset](https://mne.tools/dev/overview/datasets_index.html#somatosensory) (figures 5, A.3, A.4 in paper)

```shell
python eval_somato.py
```

> When run in parallel with 40 CPU, takes approximately
