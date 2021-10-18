# %%
import numpy as np
from scipy.stats import truncnorm

from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity

# %%


def test_truncnormkernel():
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = TruncNormKernel(lower, upper, m, sigma)

    # test get_max()
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    assert kernel.max == truncnorm.pdf(m, a, b, loc=m, scale=sigma)

    m = lower - 1
    kernel.m = m
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    assert kernel.max == truncnorm.pdf(lower, a, b, loc=m, scale=sigma)

    m = upper + 1
    kernel.m = m
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    assert kernel.max == truncnorm.pdf(upper, a, b, loc=m, scale=sigma)

    m = (lower + upper) / 2
    kernel.m = m
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    assert kernel.max == truncnorm.pdf(m, a, b, loc=m, scale=sigma)

    # test integrate()
    assert kernel.integrate(lower, upper) == 1
    assert kernel.integrate(lower - 1, lower) == 0
    assert kernel.integrate(upper, upper + 1) == 0
    assert kernel.integrate((m + lower) / 2, (m + upper) / 2) < 1

    # test call output shape
    t = 200e-3
    assert kernel(t).shape == ()

    t = [0, m, 1]
    assert kernel(t).shape == (3,)

    t = np.array([[0, m, 1],
                  [100e-3, 400e-3, 1]])
    assert kernel(t).shape == (2, 3)


def test_intensity():

    # ===== NORMAL CASES =====

    # 1d
    # define one kernel function
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = TruncNormKernel(lower, upper, m, sigma)
    driver_tt = [3.4, 5, 8, 10]  # already sorted
    # define intensity function
    baseline, alpha = 0.8, 1.2
    acti_tt = [1.2, 3, 3.6, 3.7, 4.7, 5.24, 5.5]
    intensity = Intensity(baseline, alpha, kernel, driver_tt, acti_tt)

    # test __init__()
    # XXX : test get_driver_delays

    # test get_max()
    # print(intensity.get_max())
    # print(baseline + alpha * kernel.max)
    # assert intensity.get_max() == baseline + alpha * kernel.max
    np.testing.assert_almost_equal(intensity.get_max(), baseline +
                                   alpha * kernel.max, decimal=2)

    # test __call___()
    assert intensity(0) == baseline  # before activation
    assert intensity(1.2) == baseline  # at an activation
    assert intensity(3.6) == baseline + alpha * kernel(0.2)  # on a kernel

    # 2d
    # define 2 kernel functions
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

    # test __init__()
    # XXX : test get_driver_delays

    # test __call___()
    assert intensity(0) == baseline  # before activation
    assert intensity(1.2) == baseline  # at an activation
    assert intensity(3.6) == baseline + alpha[0] * kernel[0](0.2)
    assert intensity(4.6) == baseline + alpha[1] * kernel[1](0.6)

    # 2d lifting non-overlapping assumption
    driver_tt = [[3.4, 5, 5.1, 8, 10],
                 [0.5, 2, 3.6, 8.4, 9, 10.1]]
    intensity.driver_tt = driver_tt  # update driver_tt

    tt = [0, 0.7, 3.8, 10.4]
    res = [baseline,
           baseline + alpha[1] * kernel[1](0.2),
           baseline + alpha[0] * kernel[0](0.4) + alpha[1] * kernel[1](0.2),
           baseline + alpha[0] * kernel[0](0.4) + alpha[1] * kernel[1](0.3)]
    np.testing.assert_allclose(intensity(tt), res)


def test_intensity_get_max():

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

    intensity = Intensity(baseline, alpha, kernel, driver_tt)
    assert np.all(np.around(intensity(xx), decimals=6) <= intensity.get_max())


if __name__ == '__main__':
    test_truncnormkernel()
    test_intensity()
