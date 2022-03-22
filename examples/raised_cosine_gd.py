# %%

from scipy.optimize import check_grad
import numpy as np
import warnings
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.model import Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood
from dripp.trunc_norm_kernel.optim import initialize, em_truncated_norm

EPS = 1e-5  # np.finfo(float).eps


class RaisedCosineKernel():
    """Class for raised cosine distribution kernel.

    Parameters
    ----------
    m, sigma : float | None
        Mean and standard deviation of the distribution.


    sfreq : int | None
        Sampling frequency used to create a grid between lower and upper to
        pre-compute kernel's values. If None, the kernel will be exactly
        evaluate at each call. Warning: setting sfreq to None may considerably
        increase computational time. Defaults to 150.
    """

    def __init__(self, m=1, sigma=0.2, sfreq=150.):

        if not sigma > 0:
            warnings.warn(
                "Sigma must be strictly positive, got sigma = %.3f." % sigma)

        self.sfreq = sfreq
        self.update(m=m, sigma=sigma)
        self.name = 'Raised Cosine'

    def eval(self, x):
        x = np.asarray(x)
        mask_kernel = (x < self.lower) | (x > self.upper)
        out = (1 + np.cos((x-self.m) / self.sigma * np.pi)) / \
            (2 * self.sigma)
        out[mask_kernel] = 0
        return out

    def update(self, m, sigma):
        """Update kernel with new values for shape parameters."""
        self._m = m
        self._sigma = sigma

        self.lower, self.upper = self.m - self.sigma, self.m + self.sigma

        if self.sfreq is None:
            self._x_grid = None
            self._pdf_grid = None
        else:
            # define a grid on which the kernel will be evaluate
            x_grid = np.arange(0, (self.upper - self.lower)
                               * self.sfreq + 1) / self.sfreq
            x_grid += self.lower
            self._x_grid = x_grid
            # evaluate the kernel on the pre-defined grid and save as argument
            self._pdf_grid = self.eval(x_grid)

        # compute maximum of the kernel
        self.max = self.get_max()

    def __call__(self, x):
        """Evaluate the kernel at given value(s).

        Parameters
        ----------
        x : int | float | array-like
            Value(s) to evaluate the kernel at.

        Returns
        -------
        float | numpy.array
        """

        if self.sfreq is None:
            return eval(self, x)

        x = np.atleast_1d(x)
        x_idx = np.rint((x - self.lower) * self.sfreq).astype(int)
        mask = ~np.isnan(x)
        out = np.full_like(x, fill_value=np.nan)
        mask_kernel = (x < self.lower) | (x > self.upper)
        out[mask_kernel] = 0.
        mask &= ~mask_kernel
        out[mask] = self._pdf_grid[x_idx[mask]]
        return out

    def get_max(self):
        """Compute maximum value reached by the kernel function.

        Returns
        -------
        float
        """

        return 1 / self.sigma

    def plot(self, xx=None):
        """Plot kernel.

        Parameters
        ----------
        xx : array-like | None
            Points to plot the kernel over. If None, kernel will be plotted
            over its support. Defaults to None.
        """

        if xx is None:
            x_min = max(np.floor(self.m - self.sigma), 0)
            x_max = np.ceil(self.m + self.sigma)
            xx = np.linspace(x_min, x_max, 600)

        plt.plot(xx, self.eval(xx))
        plt.xlabel('Time (s)')
        plt.title("Raised Cosine Kernel")
        plt.show()

    def partial_x(self, x):
        """
        x : array-like
        """
        x = np.atleast_1d(x)
        res = - np.pi / (2 * self.sigma) * \
            np.sin((x-self.m) / self.sigma * np.pi)

        return res

    def partial_m(self, x):
        """
        x : array-like
        """
        x = np.atleast_1d(x)
        mask_kernel = (x < self.lower) | (x > self.upper)
        res = np.pi / (2 * self.sigma) * \
            np.sin((x-self.m) / self.sigma * np.pi)
        res[mask_kernel] = 0.

        return res

    def partial_u(self, x):
        """ u = m - sigma : kernel.lower
        x : array-like
        """
        x = np.atleast_1d(x)
        mask_kernel = (x < self.lower) | (x > self.upper)
        res = np.pi / (2 * self.sigma) * \
            np.sin((x - self.lower) / self.sigma * np.pi - np.pi)
        res[mask_kernel] = 0.

        return res

    def partial_sigma(self, x):
        """
        x : array-like
        """
        x = np.atleast_1d(x)
        mask_kernel = (x < self.lower) | (x > self.upper)
        res = (self.m - x) / (2 * self.sigma**3) * np.pi * \
            np.sin((x-self.m) / self.sigma * np.pi) - self.eval(x)
        res[mask_kernel] = 0.

        return res

    def grad(self, x):
        return [self.partial_m(x), self.partial_sigma(x)]

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self.update(m=self.m, sigma=value)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self.update(m=value, sigma=self.sigma)


# %% Check grad for kernel function
kernel = RaisedCosineKernel()


def func_kernel_rc(x):
    t, m, sigma = x
    kernel = RaisedCosineKernel(m, sigma)
    return kernel(t)


def grad_kernel_rc(x):
    t, m, sigma = x
    kernel = RaisedCosineKernel(m, sigma)
    return [kernel.partial_x(t), kernel.partial_m(t)[0], kernel.partial_sigma(t)[0]]


check_grad(func_kernel_rc, grad_kernel_rc, [0.2, 0, 0.1])
# XXX why not a float

# %%


def partial_nll_rc_baseline(intensity, T):
    """

    """
    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    return T - (1 / intensity_at_t).sum()


def partial_nll_rc_alpha(intensity):
    """

    """
    res = []

    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    n_drivers_tt = np.array(
        [this_driver_tt.size for this_driver_tt in intensity.driver_tt])

    for p, delays in enumerate(intensity.driver_delays):
        kernel = intensity.kernel[p]
        val = delays.copy()
        val.data = kernel(val.data)
        this_res = np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t,
                             axis=0)

        res.append(n_drivers_tt[p] - this_res)

    return np.array(res)


def partial_nll_rc_m(intensity):
    """

    """
    res = []

    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    for p, delays in enumerate(intensity.driver_delays):
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
        val = delays.copy()
        val.data = kernel.partial_m(val.data)
        this_res = -1 * alpha * \
            np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t, axis=0)

        res.append(this_res)

        # val.data = np.sin((val.data - kernel.m) / kernel.sigma * np.pi)
        # this_res = np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t,
        #                      axis=0)
        # this_res *= -1 * alpha * np.pi / (2 * kernel.sigma**2)
        # res.append(this_res)

    return np.array(res)


def partial_nll_rc_u(intensity):
    """

    """
    res = []

    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    for p, delays in enumerate(intensity.driver_delays):
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
        val = delays.copy()
        val.data = kernel.partial_u(val.data)
        this_res = -1 * alpha * \
            np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t, axis=0)

        res.append(this_res)

    return np.array(res)


def partial_nll_rc_sigma(intensity):
    """

    """
    res = []

    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    for p, delays in enumerate(intensity.driver_delays):
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
        val = delays.copy()
        val.data = kernel.partial_sigma(val.data)
        this_res = -1 * alpha * \
            np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t, axis=0)

        res.append(this_res)

        # m, sigma = kernel.m, kernel.sigma
        # val = delays.copy()
        # temp = (val.data - m) / sigma * np.pi
        # val.data = temp / (2 * sigma**2) * np.sin(temp) - \
        #     kernel(val.data) / sigma
        # this_res = np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t,
        #                      axis=0)
        # this_res *= -1 * alpha
        # res.append(this_res)

    return np.array(res)


def gd(intensity, T, step, n_iter):
    """

    """
    hist_loss = [negative_log_likelihood(intensity, T)]
    for i in range(n_iter):
        baseline, alpha, kernel = intensity.baseline, intensity.alpha, intensity.kernel
        m, sigma = [], []
        for this_kernel in kernel:
            m.append(this_kernel.m)
            sigma.append(this_kernel.sigma)
        m = np.array(m)
        sigma = np.array(sigma)
        u = m - sigma
        # compute new values
        baseline -= step * partial_nll_rc_baseline(intensity, T)
        alpha -= step * partial_nll_rc_alpha(intensity)
        # force alpha to stay non-negative
        alpha = alpha.clip(min=0)  # project on R+
        u -= step * partial_nll_rc_u(intensity)
        u = u.clip(min=0)  # project on R+ such that m - sigma > 0
        sigma -= step * partial_nll_rc_sigma(intensity)
        sigma = sigma.clip(min=EPS)  # project on [EPS=10e-5, +inf] (def.)
        # update intensity
        m = u + sigma
        intensity.update(baseline, alpha, m, sigma)
        if(alpha.max() == 0):  # all alphas are zero
            print("alpha is null after %i iteration(s)." % i)
            break
        else:
            hist_loss.append(negative_log_likelihood(intensity, T))

    params = baseline, alpha, m, sigma
    return params, hist_loss

# %% check grad for nll with RC kernel


def func_nll_rc(x, driver_tt, acti_tt, T):
    baseline, alpha, m, sigma = x
    alpha = [alpha]
    m = [m]
    sigma = [sigma]

    intensity = Intensity(kernel=RaisedCosineKernel(),
                          driver_tt=driver_tt,
                          acti_tt=acti_tt)
    intensity.update(baseline, alpha, m, sigma)
    return negative_log_likelihood(intensity, T)


def grad_nll_rc(x, driver_tt, acti_tt, T):
    baseline, alpha, m, sigma = x
    alpha = [alpha]
    m = [m]
    sigma = [sigma]

    intensity = Intensity(kernel=RaisedCosineKernel(),
                          driver_tt=driver_tt,
                          acti_tt=acti_tt)
    intensity.update(baseline, alpha, m, sigma)
    res = [partial_nll_rc_baseline(intensity, T),
           partial_nll_rc_alpha(intensity)[0],
           -1 * partial_nll_rc_m(intensity)[0],
           partial_nll_rc_sigma(intensity)[0]]
    return res


baseline_true, alpha_true, m_true, sigma_true = 0.8, 1, 400e-3, 0.1
true_params = baseline_true, alpha_true, m_true, sigma_true
# simulate data
T = 240
driver_tt, acti_tt, _, _ = simulate_data(
    lower=0, upper=0.8, m=m_true, sigma=sigma_true, sfreq=150., baseline=baseline_true,
    alpha=alpha_true, T=T, isi=1, add_jitter=False, n_tasks=0.8, n_drivers=1,
    seed=None, return_nll=False, verbose=False)

print('Check likelihood gradient')
test_params = 0.8, 1, 200e-3, 0.1
check_grad(func_nll_rc, grad_nll_rc, test_params,
           driver_tt, acti_tt, T)


# %%
baseline_true, alpha_true, m_true, sigma_true = 0.8, [1], [400e-3], [0.1]
true_params = baseline_true, alpha_true, m_true, sigma_true

kernel_true = RaisedCosineKernel(m=m_true[0], sigma=sigma_true[0])
print('Raised cosine kernel wwith true parameters')
kernel_true.plot()

T = 240
# simulate data
driver_tt, acti_tt, kernel_tg, intensity_tg = simulate_data(
    lower=0, upper=0.8, m=m_true, sigma=sigma_true, sfreq=150., baseline=baseline_true,
    alpha=alpha_true, T=T, isi=1, add_jitter=False, n_tasks=0.8, n_drivers=1,
    seed=None, return_nll=False, verbose=False)
print('Truncated Gaussian kernel wwith initial parameters')
kernel_tg[0].plot()


# %%
intensity_rc = Intensity(kernel=RaisedCosineKernel(),
                         driver_tt=driver_tt, acti_tt=acti_tt)
# compute initial values with TG kernel
# XXX not ideal, find a init method for RC kernel
init_params = initialize(intensity_tg, T, initializer='smart_start')
print("Initials parameters:\n(mu, alpha, m, sigma) = ", init_params)
baseline_init, alpha_init, m_init, sigma_init = init_params
intensity_rc.update(baseline_true, alpha_true, m_true, sigma_true)
print('Raised cosine kernel wwith initial parameters')
intensity_rc.kernel[0].plot()

# nll_rc = negative_log_likelihood(intensity_rc, T)
# nll_tg = negative_log_likelihood(intensity_tg, T)
# print('Loss with Raised Cosine:', nll_rc)
# print('Loss with Truncated Gaussian:', nll_tg)

print("Run GD on intensity with raised cosine kernel")
params, hist_loss = gd(intensity_rc, T, 1e-4, 1000)
print("Estimated parameters:\n(mu, alpha, m, sigma) = ", params)
plt.plot(hist_loss)
plt.show()
# %%
# check with EM on TG
# res_params_em, history_params, hist_loss_em = em_truncated_norm(
#     acti_tt, driver_tt=driver_tt, lower=0, upper=0.8, T=T, sfreq=150.,
#     initializer='smart_start', alpha_pos=True, n_iter=80, verbose=False,
#     disable_tqdm=False, compute_loss=True)
# baseline_hat, alpha_hat, m_hat, sigma_hat = res_params_em
# intensity_tg.update(baseline_hat, alpha_hat, m_hat, sigma_hat)
# intensity_tg.kernel[0].plot()
# plt.plot(hist_loss_em)
# plt.show()
# %%
