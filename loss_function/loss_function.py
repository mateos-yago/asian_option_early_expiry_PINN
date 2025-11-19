from points_sampling import points_sampling
from config.training_hyperparams_config import *
from config.training_data_config import *
from derivatives.derivatives import derivatives_V
import torch
from utils.utils import asian_payoff


def pinn_loss(value_net, boundary_net, device):
    # 1) PDE residual in continuation region
    t_pde, x_pde, r_pde, sigma_pde, T_pde = points_sampling.sample_pde_points(n_pde_samples_per_batch, boundary_net, device)
    V_pde, V_t_pde, V_x_pde, V_xx_pde = derivatives_V(t_pde, x_pde, r_pde, sigma_pde, T_pde, value_net)

    pde_residual = V_t_pde + (1.0 - r_pde * x_pde) * V_x_pde \
                   + 0.5 * sigma_pde**2 * x_pde**2 * V_xx_pde
    loss_pde = torch.mean(pde_residual**2)

    # 2) Free boundary: value matching + smooth pasting
    t_fb, x_fb, r_fb, sigma_fb, T_fb = points_sampling.sample_free_boundary_points(n_boundary_samples, boundary_net, device)
    V_fb, _, V_x_fb, _ = derivatives_V(t_fb, x_fb, r_fb, sigma_fb, T_fb, value_net)

    payoff_fb = asian_payoff(t_fb, x_fb)
    loss_vm = torch.mean((V_fb - payoff_fb)**2)

    t_safe_fb = torch.clamp(t_fb, min=t_min)
    target_dx = -1.0 / t_safe_fb
    loss_sp = torch.mean((V_x_fb - target_dx)**2)

    # 3) Stopping region: x <= b(t) => V = payoff
    t_stop, x_stop, r_stop, sigma_stop, T_stop = points_sampling.sample_stopping_points(n_stopping_samples, boundary_net, device)
    V_stop, _, _, _ = derivatives_V(t_stop, x_stop, r_stop, sigma_stop, T_stop, value_net)
    payoff_stop = asian_payoff(t_stop, x_stop)
    loss_stop = torch.mean((V_stop - payoff_stop)**2)

    # 4) Terminal condition t = T
    t_term, x_term, r_term, sigma_term, T_term = points_sampling.sample_terminal_points(n_terminal_samples, device)
    V_term, _, _, _ = derivatives_V(t_term, x_term, r_term, sigma_term, T_term, value_net)
    payoff_term = asian_payoff(t_term, x_term)
    loss_term = torch.mean((V_term - payoff_term)**2)

    # 5) Far boundary x = x_max, V ~ 0
    t_far, x_far, r_far, sigma_far, T_far = points_sampling.sample_far_boundary_points(n_far_samples, device)
    V_far, _, _, _ = derivatives_V(t_far, x_far, r_far, sigma_far, T_far, value_net)
    loss_far = torch.mean(V_far**2)

    loss_total = (w_pde * loss_pde +
                  w_vm  * loss_vm  +
                  w_sp  * loss_sp  +
                  w_stop * loss_stop +
                  w_term * loss_term +
                  w_far  * loss_far)

    return {
        "loss_total": loss_total,
        "loss_pde": loss_pde,
        "loss_vm": loss_vm,
        "loss_sp": loss_sp,
        "loss_stop": loss_stop,
        "loss_term": loss_term,
        "loss_far": loss_far
    }