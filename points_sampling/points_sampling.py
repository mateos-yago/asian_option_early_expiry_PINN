import torch
from config.training_data_config import *

def sample_params(n_samples, device):
    r = r_min + (r_max - r_min) * torch.rand(n_samples, 1, device=device)
    sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(n_samples, 1, device=device)
    T = T_min + (T_max - T_min) * torch.rand(n_samples, 1, device=device)
    return r, sigma, T

def sample_pde_points(n_samples, boundary_net, device):
    r, sigma, T = sample_params(n_samples, device)

    u_t = torch.rand(n_samples, 1, device=device)
    t   = t_min + u_t * (T - t_min)  # in [t_min, T_i]

    with torch.no_grad():
        b = boundary_net(t, r, sigma, T)    # in [0,t]

    u_x = torch.rand(n_samples, 1, device=device) ** 2
    x   = b + u_x * (x_max - b)     # x > b

    return t, x, r, sigma, T

def sample_free_boundary_points(n_samples, boundary_net, device):
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t   = t_min + u_t * (T - t_min)
    x_b = boundary_net(t, r, sigma, T)
    return t, x_b, r, sigma, T

def sample_stopping_points(n_samples, boundary_net, device):
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t   = t_min + u_t * (T - t_min)

    with torch.no_grad():
        b = boundary_net(t, r, sigma, T)

    u_x = torch.rand(n_samples, 1, device=device)
    x   = u_x * b   # x in [0, b(t)]

    return t, x, r, sigma, T

def sample_terminal_points(n_samples, device):
    r, sigma, T = sample_params(n_samples, device)
    t = T.clone()  # t = T_i
    x = x_min + (x_max - x_min) * torch.rand(n_samples, 1, device=device)
    return t, x, r, sigma, T


def sample_far_boundary_points(n_samples, device):
    r, sigma, T = sample_params(n_samples, device)
    u_t = torch.rand(n_samples, 1, device=device)
    t   = t_min + u_t * (T - t_min)
    x   = torch.full_like(t, x_max)
    return t, x, r, sigma, T