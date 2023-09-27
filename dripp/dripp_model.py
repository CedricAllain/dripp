import numpy as np

from dripp.trunc_norm_kernel.optim import em_truncated_norm
from dripp.trunc_norm_kernel.metric import negative_log_likelihood
from dripp.trunc_norm_kernel.model import TruncNormKernel, Intensity


class DriPP:
    def __init__(
        self,
        lower=30e-3,
        upper=500e-3,
        sfreq=150.0,
        use_dis=True,
        init_params=None,
        initializer="moment_matching",
        alpha_pos=True,
        n_iter=80,
        verbose=False,
        disable_tqdm=False,
        compute_loss=False,
    ):
        self.lower = lower
        self.upper = upper
        self.sfreq = sfreq
        self.use_dis = use_dis
        self.init_params = init_params
        self.initializer = initializer
        self.alpha_pos = alpha_pos
        self.n_iter = n_iter
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.compute_loss = compute_loss

    def fit(self, acti_tt, driver_tt=None, T=None):
        """

        Parameters
        ----------
        acti_tt : 1d-array

        driver_tt : 2d-array like object

        T : float

        Returns
        -------
        """

        if isinstance(driver_tt, dict):
            tt = []
            labels = []
            for label, this_driver_tt in driver_tt.items():
                labels.append(label)
                tt.append(this_driver_tt)
        else:
            tt = driver_tt

        if acti_tt.dtype == np.float_:
            self.params_, self.history_params_ = em_truncated_norm(
                acti_tt,
                tt,
                self.lower,
                self.upper,
                T,
                self.sfreq,
                self.use_dis,
                self.init_params,
                self.initializer,
                self.alpha_pos,
                self.n_iter,
                self.verbose,
                self.disable_tqdm,
                self.compute_loss,
            )
            baseline, alpha, m, sigma = self.params_
            self.dict_params_ = dict(baseline=baseline, alpha=alpha, m=m, sigma=sigma)
        elif acti_tt.dtype == "O":
            raise TypeError("`acti_tt` must be 1-dimensional array-like")

        return self

    def cost(self, acti_tt, driver_tt=None, T=None):
        if T is None:
            T = max(max(tt) for tt in acti_tt) + 1

        baseline_hat, alpha_hat, m_hat, sigma_hat = self.params_
        n_drivers = len(driver_tt)
        kernel = [
            TruncNormKernel(self.lower, self.upper, self.sfreq, self.use_dis)
            for _ in range(n_drivers)
        ]
        intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
        intensity.update(baseline_hat, alpha_hat, m_hat, sigma_hat)

        loss = negative_log_likelihood(intensity, T)
        return loss
