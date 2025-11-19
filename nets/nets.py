import torch
import torch.nn as nn
from config.training_data_config import *


def build_mlp(input_dim, output_dim, hidden_layers=[64, 64, 64], activation=nn.Tanh):
    layers = []
    dims = [input_dim] + hidden_layers
    for i in range(len(hidden_layers)):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_layers[-1], output_dim))
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # inputs: t, x, r, sigma, T  -> 5 dims
        self.net = build_mlp(input_dim=5, output_dim=1)

    def forward(self, t, x, r, sigma, T):
        # normalize inputs
        t_scaled     = (t - t_min) / (T_max - t_min)
        x_scaled     = (x - x_min) / (x_max - x_min)
        r_scaled     = (r - r_min) / (r_max - r_min)
        sigma_scaled = (sigma - sigma_min) / (sigma_max - sigma_min)
        T_scaled     = (T - T_min) / (T_max - T_min)

        inp = torch.cat([t_scaled, x_scaled, r_scaled, sigma_scaled, T_scaled], dim=1)
        return self.net(inp)


class BoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # inputs: t, r, sigma, T -> 4 dims
        self.net = build_mlp(input_dim=4, output_dim=1)

    def forward(self, t, r, sigma, T):
        t_scaled     = (t - t_min) / (T_max - t_min)
        r_scaled     = (r - r_min) / (r_max - r_min)
        sigma_scaled = (sigma - sigma_min) / (sigma_max - sigma_min)
        T_scaled     = (T - T_min) / (T_max - T_min)

        inp = torch.cat([t_scaled, r_scaled, sigma_scaled, T_scaled], dim=1)
        raw = self.net(inp)
        s   = torch.sigmoid(raw)  # (0,1)
        b   = s * t               # b(t;r,σ,T) in [0, t]
        return b