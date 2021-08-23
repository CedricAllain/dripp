# %%
import numpy as np

from dripp.trunc_norm_kernel.optim import initialize_baseline, initialize
# %%


def test_initialize_baseline():
    """
    XXX
    """
    # %%
    lower = 30e-3
    upper = 500e-3
    T = 20
    acti_tt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 19])
    driver_tt = np.array([[2.5, 5.8, 7.6]], dtype='object')

    true_baseline_init = (len(acti_tt) - 3) / \
        (T - 3 * (upper - lower))

    assert initialize_baseline(
        acti_tt, driver_tt, lower, upper, T) == true_baseline_init
    # %%

    driver_tt = np.array([np.array([2.5, 5.8, 7.6]),
                          np.array([11.6, 18.6])], dtype='object')
    true_baseline_init = (len(acti_tt) - 5) / \
        (T - 5 * (upper - lower))
    # %%
    baseline_init = initialize_baseline(acti_tt, driver_tt, lower, upper, T)
    np.testing.assert_almost_equal(baseline_init,
                                   true_baseline_init,
                                   decimal=5)


def test_initialize():
    """
    XXX
    """
    # %%
    lower = 30e-3
    upper = 500e-3
    T = 20
    acti_tt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 19])
    driver_tt = np.array([np.array([2.5, 5.8, 7.6]),
                          np.array([11.6, 18.6])], dtype='object')
    _, alpha_init, m_init, sigma_init = initialize(
        acti_tt, driver_tt, lower, upper, T, initializer='smart_start', seed=None)
    true_alpha_init = [1, 1]
    np.testing.assert_allclose(alpha_init, true_alpha_init)
    true_m_init = [np.mean([0.5, 0.2, 0.4]), np.mean([0.4, 0.4])]
    np.testing.assert_allclose(m_init, true_m_init)
    true_sigma_init = [np.std([0.5, 0.2, 0.4]), np.std([0.4, 0.4])]
    np.testing.assert_allclose(sigma_init, true_sigma_init)

    # %%


if __name__ == '__main__':
    test_initialize_baseline()
