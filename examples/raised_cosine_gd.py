# %%
from cgitb import reset
from scipy.optimize import check_grad
import numpy as np
import warnings
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.simu import simulate_data
from dripp.trunc_norm_kernel.model import Intensity
from dripp.trunc_norm_kernel.metric import negative_log_likelihood
from dripp.trunc_norm_kernel.optim import initialize


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

    def partial_m(self, x):
        """
        x : array-like
        """
        x = np.asarray(x)
        res = np.pi / (2 * self.sigma) * \
            np.sin((x-self.m) / self.sigma * np.pi)

        return res

    def partial_sigma(self, x):
        """
        x : array-like
        """
        x = np.asarray(x)
        res = (x - self.m) / (2 * self.sigma**3) * np.pi * \
            np.sin((x-self.m) / self.sigma * np.pi) - self.eval(x)

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


def partial_nll_rc_baseline(intensity, T):
    """

    """
    # compute value of intensity function at each activation time
    intensity_at_t = intensity(t=intensity.acti_tt,
                               driver_delays=intensity.driver_delays)

    return T - (1 / intensity_at_t).sum() * intensity.baseline


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
        alpha, kernel = intensity.alpha[p], intensity.kernel[p]
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
        val.data = np.sin((val.data - kernel.m) / kernel.sigma * np.pi)
        this_res = np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t,
                             axis=0)
        this_res *= alpha * np.pi / (2 * kernel.sigma**2)

        res.append(-this_res)

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
        m, sigma = kernel.m, kernel.sigma
        val = delays.copy()
        temp = (val.data - m) / sigma * np.pi
        val.data = temp / (2 * sigma**2) * np.sin(temp) - \
            kernel(val.data) / sigma
        this_res = np.nansum(np.array(val.sum(axis=1).T)[0] / intensity_at_t,
                             axis=0)
        this_res *= alpha
        res.append(-this_res)

    return np.array(res)


def gd(intensity, T, step, n_iter):
    """

    """

    for i in range(n_iter):
        baseline, alpha, kernel = intensity.baseline, intensity.alpha, intensity.kernel
        m, sigma = [], []
        for this_kernel in kernel:
            m.append(this_kernel.m)
            sigma.append(this_kernel.sigma)
        m = np.array(m)
        sigma = np.array(sigma)
        # compute new values
        baseline -= step * partial_nll_rc_baseline(intensity, T)
        alpha -= step * partial_nll_rc_alpha(intensity)
        m -= step * partial_nll_rc_m(intensity)
        sigma -= step * partial_nll_rc_sigma(intensity)
        # update intensity
        intensity.update(baseline, alpha, m, sigma)

    return baseline, alpha, m, sigma


# %%
baseline_true, alpha_true, m_true, sigma_true = 0.8, 1, [500e-3], [0.1]
true_params = baseline_true, alpha_true, m_true, sigma_true
kernel_true = RaisedCosineKernel(m=500e-3, sigma=0.2)
kernel_true.plot()

T = 240
driver_tt, acti_tt, kernel_tg, intensity_tg = simulate_data(
    lower=0, upper=1, m=m_true, sigma=sigma_true, sfreq=150., baseline=baseline_true,
    alpha=alpha_true, T=T, isi=1, add_jitter=False, n_tasks=0.8, n_drivers=1,
    seed=None, return_nll=False, verbose=False)
kernel_tg[0].plot(xx=np.linspace(0, 1, 600))


# %%
intensity = Intensity(kernel=RaisedCosineKernel(),
                      driver_tt=driver_tt, acti_tt=acti_tt)
# compute initial values with TG kernel
# XXX not ideal, find a init method for RC kernel
init_params = initialize(intensity_tg, T, initializer='smart_start')
print("Initials parameters:\n(mu, alpha, m, sigma) = ", init_params)
baseline_init, alpha_init, m_init, sigma_init = init_params
intensity.update(baseline_init, alpha_init, m_init, sigma_init)

nll = negative_log_likelihood(intensity, T)
nll_tg = negative_log_likelihood(intensity_tg, T)
# %%
gd(intensity, T, 0.04, 100)
