import torch
import torch.nn as nn
from typing import Dict, Any


def get_activation(act):
    if act == 'tanh':
        act = nn.Tanh()
    elif act == 'relu':
        act = nn.ReLU()
    elif act == 'softplus':
        act = nn.Softplus()
    elif act == 'rrelu':
        act = nn.RReLU()
    elif act == 'leakyrelu':
        act = nn.LeakyReLU()
    elif act == 'elu':
        act = nn.ELU()
    elif act == 'selu':
        act = nn.SELU()
    elif act == 'glu':
        act = nn.GLU()
    else:
        print('Defaulting to tanh activations...')
        act = nn.Tanh()
    return act


def get_device(device: str = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    return device


def update_configs_with_args(configs, args_dict: Dict[str, Any], suffix):
    for key, value in args_dict.items():
        if suffix is not None:
            if key.endswith(suffix):
                # Remove the suffix
                config_key = key[:-len(suffix)]
                # Only update if the argument is provided and valid
                if hasattr(configs, config_key) and value is not None:
                    setattr(configs, config_key, value)