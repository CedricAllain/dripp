import setuptools

setuptools.setup(
    name="dripp_neurips_2021",
    version="0.0.1",
    description="Official implementation of DriPP: Driven Point Processes to "
    "Model Stimuli Induced Patterns in M/EEG Signals.",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'mne',
        'numpy',
        'pandas',
        'scipy',
        'joblib',
        'matplotlib',
    ],
)
