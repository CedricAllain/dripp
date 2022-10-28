"""Define kernel and intensity classes.
"""

import numpy as np
import itertools
import math
import numbers
from scipy.stats import truncnorm
import warnings
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.utils import \
    convert_variable_multi, get_driver_delays, check_truncation_values, check_driver_tt, check_acti_tt


class TruncNormKernel():
    """Class for truncated normal distribution kernel.

    Parameters
    ----------
    lower, upper : float
        Kernel's truncation values, thus defining its support. Defaults to
        lower = 0 and upper = 1.

    m, sigma : float | None
        Mean and standard deviation of the distribution. If None, the mean is
        set to the middle of the supper, and the standard deviation is set such
        that 95% of the kernel mass is in [m +- 2*sigma]. Defaults to None.

    sfreq : int | None
        Sampling frequency used to create a grid between lower and upper to
        pre-compute kernel's values. If None, the kernel will be exactly
        evaluate at each call. Warning: setting sfreq to None may considerably
        increase computational time. Defaults to 150.
    """

    def __init__(self, lower=0, upper=1, m=None, sigma=None, sfreq=150.):

        check_truncation_values(lower, upper)

        # compute default values of shape parameters
        if m is None:
            m = (upper - lower) / 2
        if sigma is None:
            sigma = 0.95 * (upper - lower) / 4

        if not sigma > 0:
            warnings.warn(
                "Sigma must be strictly positive, got sigma = %.3f." % sigma)

        self.lower = lower
        self.upper = upper
        self.sfreq = sfreq
        self.update(m=m, sigma=sigma)

    def update(self, m, sigma):
        """Update kernel with new values for shape parameters."""
        self._m = m
        self._sigma = sigma

        # normalize truncation values
        self._a = a = (self.lower - self.m) / self.sigma
        self._b = b = (self.upper - self.m) / self.sigma

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
            self._pdf_grid = truncnorm.pdf(
                x_grid, a, b, loc=self.m, scale=self.sigma)

        # compute maximum of the kernel
        self.max = self.get_max()

    def eval(self, x):
        """Exactly evaluate the kernel at given value(s)."""
        return truncnorm.pdf(x, self._a, self._b, loc=self.m, scale=self.sigma)

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

        x = np.asarray(x)
        x_idx = np.asarray(((x - self.lower) * self.sfreq), dtype=int)
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

        if self.m < self.lower:
            return self.eval(self.lower)
        elif self.m > self.upper:
            return self.eval(self.upper)
        else:
            return self.eval(self.m)

    def ppf(self, q):
        """Percent point function (inverse of `cdf`) at q.

        Parameters
        ----------
        q : float | array-like
            The values to compute the ppf.
        """

        return truncnorm.ppf(q, self._a, self._b, loc=self.m, scale=self.sigma)

    def interval(self, alpha):
        """Endpoints of the range that contains alpha percent of the
        distribution.

        Parameters
        ----------
        alpha : float
            Percent of distribution.

        Returns
        -------
        Tuple of 2 floats
        """

        return truncnorm.interval(
            alpha, self._a, self._b, loc=self.m, scale=self.sigma)

    def integrate(self, b1, b2):
        """Integrate the kernel between b1 and b2.

        Parameters
        ----------
        b1, b2 : floats
            Integration bounds.

        Returns
        -------
        float
        """
        integ = truncnorm.cdf(b2, a=self._a, b=self._b,
                              loc=self.m, scale=self.sigma)
        integ -= truncnorm.cdf(b1, a=self._a, b=self._b,
                               loc=self.m, scale=self.sigma)

        return integ

    def plot(self, xx=None):
        """Plot kernel.

        Parameters
        ----------
        xx : array-like | None
            Points to plot the kernel over. If None, kernel will be plotted
            over its support. Defaults to None.
        """

        if xx is None:
            step = (self.upper - self.lower) / 20
            x_min = max(np.floor(self.lower - step), 0)
            x_max = np.ceil(self.upper + step)
            xx = np.linspace(x_min, x_max, 600)

        plt.plot(xx, self.eval(xx))
        plt.xlabel('Time (s)')
        plt.title("Truncated Gaussian Kernel")
        plt.show()

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


class Intensity():
    """Class for intensity function.

    Parameters
    ----------

    baseline : int | float
        Baseline intensity value. Defaults to 0.

    alpha : int | float | array-like
        Coefficient of influence. If multiple drivers are taken into account,
        alpha must be an array-like of length the number of kernels. Defaults
        to 0.

    kernel : instance of TruncNormKernel | array-like of TruncNormKernel | None
        The kernel function(s) to take into account. Defaults to None.

    driver_tt : array-like of shape (n_drivers, n_tt) | None
        The drivers' timestamps. Defaults to None.

    acti_tt : array-like of shape (n_tt, ) | None
        The process' activation timestamps. Defaults to None.
    """

    def __init__(self, baseline=0, alpha=0, kernel=None,
                 driver_tt=None, acti_tt=None):

        self.kernel = np.atleast_1d(kernel)  # set of kernels functions
        self.n_drivers = len(self.kernel)

        self.baseline = baseline

        # set of alpha coefficients
        self.alpha = convert_variable_multi(
            alpha, self.n_drivers, repeat=True)

        # make sure we have one alpha coefficient per kernel
        assert len(self.alpha) == len(self.kernel), \
            "alpha and kernel parameters must have the same length"
        # ensure that driver_tt is a 2d array (# 1st dim. is number of drivers)
        self._driver_tt = check_driver_tt(driver_tt)
        self._acti_tt = check_acti_tt(acti_tt)  # ensure it is numpy 1d-array

        if len(self.acti_tt) > 0 and self.n_drivers > 0:
            # for every process activation timestamps,
            # get its corresponding driver timestamp
            self.driver_delays = get_driver_delays(self, self.acti_tt)
        else:
            self.driver_delays = None

    def update(self, baseline, alpha, m, sigma):
        """Update the intensity function with new values.

        Update the full intensity object, the baseline parameter as well as
        the associated kernels (their shape parameters) and alphas coefficients
        In practice, this method is called once an interation of the learning
        algorithm is computed.

        Parameters
        ----------

        baseline : float
            The new value for the baseline intensity value.

        alpha : list of floats | array-like of floats
            The list of the new importance coefficient, one for each driver.

        m, sigma : list of floats | array-like of floats
            The list of the new values for the mean and std for every kernel.
        """

        self.baseline = baseline
        self.alpha = alpha
        for i in range(self.n_drivers):
            self.kernel[i].update(m=m[i], sigma=sigma[i])

    def __call__(self, t, driver_delays=None, intensity_pos=False):
        """Evaluate the intensity at time(s) t.

        Parameters
        ----------
        t : int | float | array-like
            The value(s) to evaluate the intensity at.

        driver_delays : list of scipy.sparse.csr_matrix
            A list of csr matrices, each one containing, for each driver,
            the delays between the intensity activations and the driver
            timestamps. Defaults to None.

        intensity_pos : bool
            If True, enforce the positivity of the intensity function by
            applying a max with 0. Defaults to False.

        Returns
        -------
        float | numpy.array
            The value of the intensity function at given time(s).
        """

        t = np.atleast_1d(t)

        # compute the driver delays if not specified
        if driver_delays is None:
            driver_delays = get_driver_delays(self, t)

        # initialize
        intensities = self.baseline
        for p, delays in enumerate(driver_delays):
            if delays.data.size == 0 or self.alpha[p] == 0:
                # no delays for this driver of alpha is 0
                continue
            val = delays.copy()
            # compute the kernel value at the "good" delays
            val.data = self.kernel[p](delays.data)
            intensities += self.alpha[p] * np.array(val.sum(axis=1).T)[0]

        intensities = np.atleast_1d(intensities)

        if t.size == 1:
            return intensities[0]

        if intensity_pos:
            return intensities.clip(min=0)
        else:
            return intensities

    def get_max(self):
        """Compute maximum intensity.

        Returns
        -------
        float
        """

        # get sfreq used for kernel initialization
        sfreq = self.kernel[0].sfreq
        # for each kernel do a convolution with its events timestamps
        intensity_grid = []
        for p in range(self.n_drivers):
            tt_idx = np.asarray((self.driver_tt[p] * sfreq), dtype=int)
            # create Dirac vector for driver events
            dirac_tt = np.zeros(tt_idx.max() + 1)
            dirac_tt[tt_idx] = 1
            # define the kernel pattern to use in convolution
            kernel = self.kernel[p]
            kernel_grid = kernel(np.linspace(
                0, kernel.upper, int(np.ceil(kernel.upper * sfreq))))
            # do the convolution
            this_intensity_grid = np.convolve(
                dirac_tt, kernel_grid, mode='full')
            # multiply by the corresponding factor
            this_intensity_grid *= self.alpha[p]
            intensity_grid.append(this_intensity_grid)

        # pad with 0 the intensity vectors
        intensity_grid = np.array(
            list(itertools.zip_longest(*intensity_grid, fillvalue=0))).T

        # sum accros the drivers
        intensity_grid = intensity_grid.sum(axis=0)
        # add the baseline intensity
        intensity_grid += self.baseline

        return intensity_grid.max()

    def plot(self, xx=np.linspace(0, 1, 600)):
        """Plot kernel.

        Parameters
        ----------

        xx : array-like
            Points to plot the intensity over. Default to
            numpy.linspace(0, 1, 600).
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
            Limits of the considered interval.

        n : int
            Number of events.

        Returns
        -------
        float
        """

        integ = self.baseline * (b-a)
        if self.alpha > 0:
            integ += self.alpha * self.kernel.integrate(a, b)

        proba = integ ** n * np.exp(integ) / math.factorial(n)

        return proba

    @ property
    def driver_tt(self):
        return self._driver_tt

    @ driver_tt.setter
    def driver_tt(self, value):
        self._driver_tt = np.array(value)
        # recompute driver delays from activation timestamps
        self.driver_delays = get_driver_delays(self, self.acti_tt)

    @ property
    def acti_tt(self):
        return self._acti_tt

    @ acti_tt.setter
    def acti_tt(self, value):
        self._acti_tt = np.array(value)
        # recompute driver delays from activation timestamps
        self.driver_delays = get_driver_delays(self, self.acti_tt)


if __name__ == '__main__':
    baseline, alpha = 1, [2, 1]
    # define 2 kernel functions
    m, sigma = 200e-3, 0.08
    lower, upper = 30e-3, 500e-3
    kernel = [TruncNormKernel(lower, upper, m, sigma),
              TruncNormKernel(lower, upper, m, sigma)]
    driver_tt = [[3.4, 5, 5.1, 8, 10],
                 [0.5, 2, 4]]
    acti_tt = [1.2, 3, 3.6, 3.7, 4.7, 5.24, 5.5]
    intensity = Intensity(baseline, alpha, kernel, driver_tt, acti_tt)
