import torch
import torch.nn as nn


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


def get_device(device: str = 'cuda:0'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    return device