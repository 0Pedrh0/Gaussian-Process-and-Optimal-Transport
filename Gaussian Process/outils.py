# Dans votre fichier outils.py, remplacez par :

import torch
import numpy as np

def Spec_Mix(x1, x2, gamma, mu, sigma):
    # Version PyTorch
    if isinstance(x1, torch.Tensor):
        return sigma**2 * torch.exp(-(gamma * outersum(x1, -x2)**2)) * torch.cos(2.0 * np.pi * mu * outersum(x1, -x2))
    # Version NumPy (fallback)
    else:
        return sigma**2 * np.exp(-(gamma * outersum(x1, -x2)**2)) * np.cos(2.0 * np.pi * mu * outersum(x1, -x2))

def outersum(a, b):
    # Version PyTorch
    if isinstance(a, torch.Tensor):
        return a.reshape(-1, 1) + b.reshape(1, -1)
    # Version NumPy (fallback)
    else:
        return np.add.outer(a, b)