# %%
import numpy as np
import matplotlib.pyplot as plt

import torch

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes
from tick.hawkes import HawkesKernelTimeFunc
from tick.hawkes import SimuInhomogeneousPoisson
from tick.plot import plot_point_process


def chek_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def raised_cosine_kernel(t, params, dt=1/1000, kernel_zero_base=False):
    """

    """
    t = chek_tensor(t)
    params = chek_tensor(params)

    _, alpha, mu, sig = params

    kernel = (1 + torch.cos((t-mu)/sig*np.pi)) / (2 * sig ** 2)
    mask_kernel = (t < (mu-sig)) | (t > (mu+sig))
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel / (kernel.sum() * dt)

    return kernel


def truncated_gaussian_kernel(t, params, lower, upper, dt=1/1000,
                              kernel_zero_base=False):
    """

    """
    t = chek_tensor(t)
    params = chek_tensor(params)

    _, alpha, mu, sig = params

    kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
    mask_kernel = (t < lower) | (t > upper)
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel / (kernel.sum() * dt)

    return kernel

# %%


def simu(true_params, simu_params=[50, 1000, 0.5], seed=None,
         plot_convolve=False):
    """

    Parameters
    ----------


    Returns
    -------
    intensity_csc

    z : array-like
        sparse vector where 1 indicates an intensity activation
    """

    mu_0, alpha_true, mu_true, sig_true = true_params
    T, L, p_task = simu_params
    dt = 1 / L

    # simulate data
    t_value = np.linspace(0, 1, L + 1)[:-1]
    y_value = np.array(raised_cosine_kernel(t_value, true_params, dt))
    # kernel = HawkesKernelTimeFunc(t_values=t_value, y_values=y_value)
    # simu = SimuHawkes([[kernel]], [mu_0], end_time=T)
    # simu.track_intensity(dt)
    # simu.simulate()

    # y_value = (1 + np.cos((t_value-mu_true)/sig_true*np.pi)) / \
    #     (2 * sig_true ** 2)
    # mask = (t_value <= (mu_true - sig_true)) | (t_value >= (mu_true + sig_true))
    # y_value[mask] = 0.
    # y_value /= (y_value.sum() * dt)
    # y_value *= alpha_true

    # y_value = np.sin(np.pi * t_value)  # sinus kernel

    # # project activation timestamps onto a grid of precision dt
    # t_k = (simu.timestamps[0] / dt).astype(int) * dt
    # # get intensity values
    # t_dt = simu.intensity_tracked_times / dt
    # intensity = simu.tracked_intensity[0][abs(t_dt - np.round(t_dt)) < 1e-6]

    isi = 0.7
    t_k = np.arange(start=0,
                    stop=T - 2 * isi,
                    step=isi)
    # sample timestamps
    rng = np.random.RandomState(seed=seed)
    t_k = rng.choice(t_k, size=int(p_task * len(t_k)),
                     replace=False).astype(float)
    t_k = (t_k / dt).astype(int) * dt

    t = np.arange(0, T + 1e-10, dt)
    driver_tt = t * 0
    driver_tt[(t_k * L).astype(int)] += 1
    intensity_csc = mu_0 + np.convolve(driver_tt, y_value)[:-L+1]

    #
    tf = TimeFunction((t, intensity_csc), dt=dt)
    # We define a 1 dimensional inhomogeneous Poisson process with the
    # intensity function seen above
    in_poi = SimuInhomogeneousPoisson([tf], end_time=T, verbose=False)
    # We activate intensity tracking and launch simulation
    in_poi.track_intensity(dt)
    in_poi.simulate()

    # We plot the resulting inhomogeneous Poisson process with its
    # intensity and its ticks over time
    plot_point_process(in_poi)

    t_k = (in_poi.timestamps[0] / dt).astype(int) * dt
    acti_tt = t * 0
    acti_tt[(t_k * L).astype(int)] += 1

    if plot_convolve:

        # z_test

        plt.plot(t, intensity_csc)
        plt.plot(t, intensity, '--')
        # plt.vlines(t_k, mu_0, intensity.max(), linestyles='dotted')
        plt.show()

    return y_value, intensity_csc, driver_tt, acti_tt, in_poi


def run_exp(true_params, simu_params=[50, 1000], init_params=None,
            loss='log-likelihood', kernel_zero_base=False, max_iter=100,
            step_size=1e-5):
    """

    Parameters
    ----------
    true_params : list
        [mu_0, alpha_true, mu_true, sig_true]

    simu_params = list
        [T, L]


    """

    # simulate data
    true_kernel, true_intensity, driver_tt, acti_tt, in_poi = simu(
        true_params, simu_params)

    # intialize parameters
    P0 = torch.tensor(init_params, requires_grad=True)
    mu0, alpha, mu, sig = P0

    dt = 1 / simu_params[1]
    t = torch.arange(0, 1, dt)

    pobj = []
    driver_torch = torch.tensor(driver_tt).float()
    driver_t = driver_torch.to(torch.bool)

    acti_torch = torch.tensor(acti_tt).float()
    acti_t = acti_torch.to(torch.bool)

    for i in range(max_iter):
        print(f"Fitting model... {i/max_iter:6.1%}\r", end='', flush=True)
        P0.grad = None
        # kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
        kernel = raised_cosine_kernel(t, P0, dt, kernel_zero_base)
        # kernel = (1 + torch.cos((t-mu)/sig*np.pi)) / (2 * sig ** 2)
        # mask_kernel = (t < (mu-sig)) | (t > (mu+sig))
        # kernel[mask_kernel] = 0.
        # if kernel_zero_base:
        #     kernel = (kernel - kernel.min())
        # kernel = alpha * kernel / (kernel.sum() * dt)

        # torch.exp(mu0)
        intensity = mu0 + torch.conv1d(driver_torch[None, None],
                                       kernel[None, None],
                                       padding=(L-1,))[0, 0, :-L+1]
        # assert np.allclose(intensity.detach(), np.convolve(z, k.detach()))
        # detach() allow to use as array
        if loss == 'log-likelihood':
            v_loss = intensity.sum() * dt - torch.log(intensity[acti_t]).sum()
        elif loss == 'MLE':
            v_loss = (intensity ** 2).sum() * dt - \
                2 * (intensity[acti_t]).sum()
        else:
            raise ValueError(
                f"loss must be 'MLE' or 'log-likelihood', got '{loss}'"
            )
        v_loss.backward()
        P0.data -= step_size * P0.grad
        P0.data[1] = max(0, P0.data[1])  # alpha
        P0.data[3] = max(0, P0.data[3])  # sigma
        P0.data[2] = max(P0.data[3], P0.data[2])  # m s.t. m - sigma > 0

        pobj.append(v_loss.item())

    print("Fitting model... done  ")
    print(f"Estimated parameters: {P0.data}")

    fig = plt.figure()
    gs = plt.GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(intensity.detach(), label="Estimated intensity")
    ax.plot(true_intensity, '--', label="True intensity")
    # plot_point_process(in_poi, ax=ax)
    ax.set_title("Intensity function")
    ax.set_xlabel('')
    # ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    # ax.set_title("Cost function")
    ax.plot(pobj, label=f"{loss}")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(kernel.detach(), label='Learned kernel')
    ax.plot(true_kernel, label='True kernel')
    ax.legend()
    plt.show()

    return P0
# %%


# true parameters
mu_0 = 0.7
alpha_true = 0.7
mu_true = 0.3
sig_true = 0.2
true_params = np.array([mu_0, alpha_true, mu_true, sig_true])
# simulation parameters
T = 100
L = 1000
p_task = 0.3
simu_params = [T, L, p_task]
# initialize parameters
rng = np.random.RandomState(seed=42)
p = 0.1  # +- p around true parameters
init_params = rng.uniform(low=true_params*(1-p), high=true_params*(1+p))

est_params = run_exp(true_params, simu_params, init_params, max_iter=200,
                     step_size=1e-4)

# %%
