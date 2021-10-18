# %%

import numpy as np
import itertools
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity


# define global grid
sfreq = 500
T = 10
xx = np.linspace(0, T, T*sfreq)
# define intensity parameters
alpha = [0.8, 1.2]
baseline = 0.8
# define kernels functions and their event timestamps
kernel_1 = TruncNormKernel(lower=30e-3, upper=800e-3,
                           m=400e-3, sigma=0.1, sfreq=sfreq)
driver_tt_1 = np.array([1, 2.4, 4, 5, 6, 7.3])
kernel_2 = TruncNormKernel(lower=0, upper=1.5,
                           m=600e-3, sigma=0.4, sfreq=sfreq)
driver_tt_2 = np.array([1.3, 2, 3, 5, 5.5, 7])

kernel = [kernel_1, kernel_2]
driver_tt = [driver_tt_1, driver_tt_2]

intensity_grid = []
for this_kernel, this_driver_tt, this_alpha in zip(kernel, driver_tt, alpha):
    tt_idx = np.asarray((this_driver_tt * sfreq), dtype=int)
    # do a convolution to obtain the intensity function
    # dirac_tt = np.zeros_like(xx)
    dirac_tt = np.zeros(tt_idx.max() + 1)
    dirac_tt[tt_idx] = 1
    kernel_grid = this_kernel(np.linspace(
        0, this_kernel.upper, this_kernel.upper*sfreq))
    this_intensity_grid = np.convolve(dirac_tt, kernel_grid, mode='full')
    this_intensity_grid *= this_alpha
    intensity_grid.append(this_intensity_grid)

# pad with 0
intensity_grid = np.array(
    list(itertools.zip_longest(*intensity_grid, fillvalue=0))).T
# sum accros the drivers
intensity_grid = intensity_grid.sum(axis=0)
intensity_grid += baseline
# get max
print("maximum intensity:", intensity_grid.max())

plt.plot(xx[:intensity_grid.shape[0]], intensity_grid[:xx.shape[0]])
plt.show()

# %%

intensity = Intensity(baseline, alpha, kernel, driver_tt)
print(intensity.get_max())

# %%
