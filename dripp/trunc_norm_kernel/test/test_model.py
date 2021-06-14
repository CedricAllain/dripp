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


def test_intensity():

    # ===== NORMAL CASES =====

    # 1d
    # define one kernel function
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = TruncNormKernel(lower, upper, m, sigma)
    driver_tt = [3.4, 5, 5.1, 8, 10]  # already sorted

    baseline, alpha = 0.8, 1.2
    acti_tt = [1.2, 3, 3.6, 3.7, 4.7, 5.24, 5.5]
    intensity = Intensity(baseline, alpha, kernel, driver_tt, acti_tt)

    # test __init__()
    acti_last_tt = np.array([[np.nan, np.nan, 3.4, 3.4, 3.4, 5.1, 5.1]])
    np.testing.assert_allclose(intensity.acti_last_tt, acti_last_tt)

    # test get_max()
    assert intensity.get_max() == baseline + alpha * kernel.max

    # test __call___()
    assert intensity(0) == baseline
    assert intensity(1.2) == baseline
    assert intensity(3.6) == baseline + alpha * kernel(0.2)


if __name__ == '__main__':
    # test_truncnormkernel()
    test_intensity()
