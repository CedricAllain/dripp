# %%
import numpy as np
import math
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.utils import get_last_timestamps


class TruncNormKernel():
    """Class for truncated normal distribution kernel

    Parameters
    ----------
    lower, upper : float
        kernel's truncation values

    m : float
        mean of the distribution

    sigma : float
        standard deviation of the distribution

    sfreq : int | None
        sampling frequency used to create a grid between lower and upper to
        pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    """

    def __init__(self, lower, upper, m, sigma, sfreq=150.):

        assert lower < upper, \
            "Truncation value 'lower' must be strictly smaller than 'upper'."

        assert sigma > 0, "Sigma must be strictly positive."

        self.lower = lower
        self.upper = upper
        self.sfreq = sfreq
        self.update(m=m, sigma=sigma)
        self.max = self.get_max()  # compute maximum of the kernel

    def update(self, m, sigma):
        self._m = m
        self._sigma = sigma

        # normalize truncation values
        self._a = a = (self.lower - self.m) / self.sigma
        self._b = b = (self.upper - self.m) / self.sigma

        if self.sfreq is None:
            self._x_grid = None
            self._pdf_grid = None
        else:
            x_grid = np.arange(0, (self.upper - self.lower)
                               * self.sfreq + 1) / self.sfreq
            x_grid += self.lower
            self._x_grid = x_grid
            self._pdf_grid = truncnorm.pdf(
                x_grid, a, b, loc=self.m, scale=self.sigma)

    @property
    def sigma(self):
        return self._sigma

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        self.max = self.get_max()  # recompute max

    def __call__(self, x):
        """Evaluate the kernel at given value(s)

        Parameters
        ----------
        x : int | float | array-like
            value(s) to evaluate the kernel at

        Returns
        -------
        float | numpy.array

        """
        if self.sfreq is None:
            return truncnorm.pdf(x, self._a, self._b, loc=self.m,
                                 scale=self.sigma)

        x = np.asarray(x)
        x_idx = np.asarray(((x - self.lower) * self.sfreq), dtype=int)
        mask = ~np.isnan(x)
        out = np.full_like(x, fill_value=np.nan)
        mask_kernel = (x < self.lower) | (x > self.upper)
        out[mask_kernel] = 0.
        mask &= ~mask_kernel
        out[mask] = self._pdf_grid[x_idx[mask]]
        return out

    def eval(self, x):
        return truncnorm.pdf(x, self._a, self._b, loc=self.m, scale=self.sigma)

    def get_max(self):
        """Compute maximum value reached by the kernel function

        Returns
        -------
        float

        """

        if self.m < self.lower:
            return self.eval(self.lower)
        elif self.m > self.upper:
            return self.eval(self.upper)
        else:
            return self.eval(self.m)

    def ppf(self, q):
        """Percent point function (inverse of `cdf`) at q

        Parameters
        ----------
        q : float | array-like
            the values to compute the ppf

        """

        # normalize truncation values
        a = (self.lower - self.m) / self.sigma
        b = (self.upper - self.m) / self.sigma

        return truncnorm.ppf(q, a, b, loc=self.m, scale=self.sigma)

    def interval(self, alpha):
        """Endpoints of the range that contains alpha percent of the
        distribution

        Parameters
        ----------
        alpha : float
            percent of distribution

        Returns
        -------
        tuple of 2 floats

        """

        # normalize truncation values
        a = (self.lower - self.m) / self.sigma
        b = (self.upper - self.m) / self.sigma

        return truncnorm.interval(alpha, a, b, loc=self.m, scale=self.sigma)

    def plot(self, xx=None):
        """Plot kernel

        Parameters
        ----------
        xx : array-like | None
            points to plot the kernel over
            if None, kernel will be plotted over its support
            default is None

        """

        if xx is None:
            step = (self.upper - self.lower) / 20
            x_min = max(np.floor(self.lower - step), 0)
            x_max = np.ceil(self.upper + step)
            xx = np.linspace(x_min, x_max, 600)

        plt.plot(xx, self.eval(xx))
        plt.xlabel('Time (s)')
        plt.title("Kernel function")
        plt.show()

    def integrate(self, b1, b2):
        """Integrate the kernel between b1 and b2

        Parameters
        ----------
        b1, b2 : floats
            integration bounds

        Returns
        -------
        float

        """

        # normalize truncation values
        a = (self.lower - self.m) / self.sigma
        b = (self.upper - self.m) / self.sigma

        integ = truncnorm.cdf(b2, a=a, b=b, loc=self.m, scale=self.sigma)
        integ -= truncnorm.cdf(b1, a=a, b=b, loc=self.m, scale=self.sigma)

        return integ

# %%


class Intensity():
    """Class for intensity function

    Parameters
    ----------

    baseline : int | float
        baseline intensity

    alpha : int | float
        coefficient of influence
        default is 0

    kernel : instance of TruncNormKernel
        kernel function

    driver_tt : array-like of shape (n_drivers, n_tt)
        the drivers' timestamps
        default is ()

    acti_tt : array-like of shape (n_tt, )
        the process' activation timestamps
        default is ()

    """

    def __init__(self, baseline, alpha=0, kernel=None,
                 driver_tt=(), acti_tt=()):

        self.baseline = baseline
        self.alpha = np.atleast_1d(alpha)  # set of alpha coefficients
        self.kernel = np.atleast_1d(kernel)  # set of kernels functions
        self.acti_tt = np.atleast_1d(acti_tt)  # ensure it is numpy array
        # ensure that driver_tt is a 2d array (# 1st dim. is # drivers)
        if isinstance(driver_tt[0], (int, float)):
            driver_tt = np.atleast_2d(driver_tt)
        self.driver_tt = np.array([np.array(x) for x in driver_tt])

        # compute maximum intensity
        self.lambda_max = self.get_max()

        if acti_tt.shape[0] > 0 and driver_tt.shape[0] > 0:
            # for every process activation timestamps,
            # get its corresponding driver timestamp
            self.acti_last_tt = get_last_timestamps(driver_tt, acti_tt)
        else:
            self.acti_last_tt = ()

    def get_max(self):
        """Compute maximum intensity

        Returns
        -------
        float

        """

        m = self.baseline
        if self.kernel.shape[0] > 0:  # if at least one associated kernel
            m += np.array([alpha * kernel.max for alpha,
                           kernel in zip(self.alpha, self.kernel)]).max()

        # if self.kernel is not None:
        #     return self.baseline + self.alpha * self.kernel.max
        # else:
        #     return self.baseline

        return m

    def __call__(self, t, last_tt=()):
        """Evaluate the intensity at time(s) t

        Parameters
        ----------
        t : int | float | array-like
            the value(s) we would like to evaluate the intensity at

        last_tt : int | float | array-like
            the corresponding driver's timestamps
            if not specified, will be computed

        Returns
        -------
        float | numpy.array
            the value of the intensity function at given time(s)

        """

        t = np.atleast_1d(t)
        last_tt = np.atleast_2d(last_tt)  # from 1d to 2d
        # with the non-overlapping assumption, only the last event on each
        # driver matters

        # if (self.alpha == 0) or (len(self.driver_tt) == 0):
        #   return np.full(t.size, fill_value=self.baseline)

        if last_tt.shape[0] != self.driver_tt.shape[0]:
            last_tt = get_last_timestamps(self.driver_tt, t)

        intensities = self.baseline
        for alpha, kernel, tt in zip(self.alpha, self.kernel, last_tt):
            # if (self.kernel is not None) and (self.alpha > 0):
            intensities += alpha * kernel(t - tt)

        intensities[np.isnan(intensities)] = self.baseline

        if t.size == 1:
            return intensities[0]

        return intensities

    def plot(self, xx=np.linspace(0, 1, 600)):
        """Plot kernel

        Parameters
        ----------

        xx : array-like
            points to plot the intensity over
            default is numpy.linspace(0, 1, 600)

        """
        yy = self.baseline
        for alpha, kernel in zip(self.alpha, self.kernel):
            yy += alpha * kernel.eval(xx)

        plt.plot(xx, yy)
        plt.xlabel('Time (s)')
        plt.title("Intensity function at kernel")
        plt.show()

    def compute_proba(self, a, b, n):
        r"""Compute the probability to have n events in the intervalle [a,b].

        The probability is given by the following formula:

        .. math::

            \frac{1}{n!} \left( \int_a^b \lambda(x) dx \right)^n \
            \exp\left(\int_a^b \lambda(x) dx\right)

        Parameters
        ----------
        a, b : floats
            limits of the considered interval

        n : int
            number of events

        Returns
        -------
        float
        """

        integ = self.baseline * (b-a)
        if self.alpha > 0:
            integ += self.alpha * self.kernel.integrate(a, b)

        proba = integ ** n * np.exp(integ) / math.factorial(n)

        return proba

    @property
    def driver_tt(self):
        return self._driver_tt

    @driver_tt.setter
    def driver_tt(self, value):
        self._driver_tt = np.array(value)

    @property
    def acti_tt(self):
        return self._acti_tt

    @acti_tt.setter
    def acti_tt(self, value):
        self._acti_tt = np.array(value)
