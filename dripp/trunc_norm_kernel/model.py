# %%
import numpy as np
import math
from scipy.stats import truncnorm
from copy import deepcopy
from joblib import Memory, Parallel, delayed
import matplotlib.pyplot as plt

from dripp.trunc_norm_kernel.utils import \
    convert_variable_multi, get_last_timestamps


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

    @sigma.setter
    def sigma(self, value):
        self.update(m=self.m, sigma=value)
        self.max = self.get_max()  # recompute max

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self.update(m=value, sigma=self.sigma)
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
        # if np.issubdtype(x[0].dtype, np.number):

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

    alpha : int | float | array-like
        coefficient of influence
        if multiple drivers are taken into account, alpha must be an array-like
        of length the number of kernels.
        default is 0

    kernel : instance of TruncNormKernel | array-like of TruncNormKernel
        the kernel function(s) to take into account
        default is None

    driver_tt : array-like of shape (n_drivers, n_tt)
        the drivers' timestamps
        default is ()

    acti_tt : array-like of shape (n_tt, )
        the process' activation timestamps
        default is ()

    """

    def __init__(self, baseline, alpha=0, kernel=None,
                 driver_tt=(), acti_tt=()):

        # ensure that driver_tt is a 2d array (# 1st dim. is # drivers)
        if isinstance(driver_tt[0], (int, float)):
            driver_tt = np.atleast_2d(driver_tt)
        self.driver_tt = np.array([np.array(x) for x in driver_tt])

        self.n_driver = len(self.driver_tt)
        self.baseline = baseline
        # set of alpha coefficients
        self.alpha = convert_variable_multi(
            alpha, len(self.driver_tt), repeat=True)
        self.kernel = np.atleast_1d(kernel)  # set of kernels functions
        self.acti_tt = np.atleast_1d(acti_tt)  # ensure it is numpy array

        # make sure we have one alpha coefficient per kernel
        assert len(self.alpha) == len(self.kernel), \
            "alpha and kernel parameters must have the same length"

        if self.acti_tt.shape[0] > 0 and self.driver_tt.shape[0] > 0:
            # for every process activation timestamps,
            # get its corresponding driver timestamp
            self.acti_last_tt = get_last_timestamps(driver_tt, acti_tt)
        else:
            self.acti_last_tt = ()

        # compute maximum intensity
        # self.lambda_max = self.get_max()

    def update(self, baseline, alpha, m, sigma):
        """Update the intensity function (baseline parameter as well as associated kernels and alpha) with new values.
        In practice, this method is called once an interation of the learning
        algorithm is computed.

        """

        self.baseline = baseline
        self.alpha = alpha
        for i in range(self.n_driver):
            self.kernel[i].update(m=m[i], sigma=sigma[i])

    def __call__(self, t, last_tt=(), non_overlapping=False):
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

        if non_overlapping:
            if (last_tt == ()) or \
                    (last_tt.shape[0] != self.driver_tt.shape[0]):
                # with the non-overlapping assumption, only the last event on
                # each driver matters
                last_tt = get_last_timestamps(self.driver_tt, t)
            else:
                last_tt = np.atleast_2d(last_tt)  # from 1d to 2d

            intensities = self.baseline
            for alpha, kernel, tt in zip(self.alpha, self.kernel, last_tt):
                # if (self.kernel is not None) and (self.alpha > 0):
                intensities += alpha * kernel(t - tt)

            intensities[np.isnan(intensities)] = self.baseline
        else:
            # get number of drivers
            n_drivers = len(self.kernel)
            # initialize
            intensities = self.baseline
            for p in range(n_drivers):
                # compute delays
                delays = t[:, np.newaxis] - self.driver_tt[p]
                # compute sum of kernel values for these delays
                val = np.nansum(self.kernel[p](delays.astype('float')), axis=1)
                intensities += self.alpha[p] * val

        if t.size == 1:
            return intensities[0]

        return intensities

    def get_max(self):
        """Compute maximum intensity

        Returns
        -------
        float

        """

        # lifting the non-overlapping assumption: get empirical max
        # first_xx = np.floor(np.array([tt.min()
        #                               for tt in self.driver_tt]).min())
        # last_xx = np.ceil(np.array([tt.max() for tt in self.driver_tt]).max())

        # xx = np.linspace(first_xx, last_xx, int(sfreq) * int(last_xx-first_xx))
        # m = self(xx).max()

        # ====================================

        # compute the intensity over all kernels' supports and get max
        # sfreq = self.kernel[0].sfreq
        # lower, upper = self.kernel[0].lower, self.kernel[0].upper
        # all_tt = np.sort(np.hstack(self.driver_tt.flatten()))
        # m = self.baseline

        # # get all supports (where intensity > baseline)
        # supports = []
        # temp = (all_tt[0] + lower, all_tt[0] + upper)
        # for i in range(all_tt.size - 1):
        #     if all_tt[i+1] + lower > temp[1]:
        #         # xx = np.linspace(
        #         #     temp[0], temp[1], int(sfreq*(temp[1]-temp[0])))
        #         # m = max(m, self(xx).max())  # update maximum value
        #         supports.append(temp)
        #         temp = (all_tt[i+1] + lower, all_tt[i+1] + upper)
        #     else:
        #         temp = (temp[0], all_tt[i+1] + upper)

        # def get_sub_max(support):
        #     """

        #     """
        #     xx = np.linspace(
        #         support[0], support[1], int(sfreq*(support[1]-support[0])))
        #     return self(xx).max()

        # all_m = Parallel(n_jobs=40)(
        #     delayed(get_sub_max)(this_support) for this_support in supports[:200])
        # m = np.max(all_m)

        # ====================================
        # compute a supremum
        m = 0
        for p in range(len(self.kernel)):
            m = max(m, self.alpha[p] * self.kernel[p].get_max())

        m += self.baseline

        # ====================================

        # if "global" (i.e., all drivers combined) non-overlapping assumption
        # m = self.baseline
        # if self.kernel.shape[0] > 0:  # if at least one associated kernel
        #     m += np.array([alpha * kernel.max for alpha,
        #                     kernel in zip(self.alpha, self.kernel)]).max()

        # if self.kernel is not None:
        #     return self.baseline + self.alpha * self.kernel.max
        # else:
        #     return self.baseline

        return m

    def get_next_lambda_max(self, t):
        """Given a point in time, compute the maximum of the intensity in the
        near future, by taking into account only past driver events.

        lambda^*(t) = max_{t'>t} lambda(t' | \mathcal{F}_t)

        Parameters
        ----------
        t : float

        Returns
        -------

        """

        xx_max = t + np.array([k.upper for k in self.kernel]).max()
        xx = np.linspace(t, xx_max, int(500 * (xx_max-t)))  # 500 points/sec
        # make a copy of the current intensity function but filter its driver
        # events
        other_intensity = deepcopy(self)
        other_intensity.driver_tt = np.array([np.array(x[x < t])
                                              for x in self.driver_tt])

        return other_intensity(xx).max()

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

    @ property
    def driver_tt(self):
        return self._driver_tt

    @ driver_tt.setter
    def driver_tt(self, value):
        self._driver_tt = np.array(value)

    @ property
    def acti_tt(self):
        return self._acti_tt

    @ acti_tt.setter
    def acti_tt(self, value):
        self._acti_tt = np.array(value)


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
