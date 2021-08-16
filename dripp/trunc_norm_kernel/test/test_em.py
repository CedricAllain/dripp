import numpy as np

from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity
from dripp.trunc_norm_kernel.em import compute_p_tp


def test_compute_p_tp():
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = [TruncNormKernel(lower, upper, m, sigma),
              TruncNormKernel(lower, upper, m, sigma)]
    driver_tt = [[3.4, 5.1, 8, 10],
                 [0.5, 2, 4]]  # make sure it respects non-overlapping
    # define intensity function
    baseline, alpha = 0.8, [1.2, 0.5]
    acti_tt = [1.2, 3, 3.6, 3.7, 4.7, 5.24, 5.5]
    intensity = Intensity(baseline, alpha, kernel, driver_tt, acti_tt)

    t = [1, 3.6, 5.5, 9]
    assert compute_p_tp(t, intensity).shape == (len(kernel), len(t))


if __name__ == '__main__':
    test_compute_p_tp()
