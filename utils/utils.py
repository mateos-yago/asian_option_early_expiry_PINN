from config.training_data_config import *
import torch

def asian_payoff(t, x):
    t_safe = torch.clamp(t, min=t_min)
    val = 1.0 - x / t_safe
    return torch.clamp(val, min=0.0)