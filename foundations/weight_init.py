import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / (fan_in+fan_out))**0.5
        W = torch.randn(fan_out, fan_in)*std
        return [[round(v.item(), 4) for v in row] for row in W]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / fan_in)**0.5
        W = torch.randn(fan_out, fan_in)*std
        return [[round(v.item(), 4) for v in row] for row in W]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        weights = []
        in_dim = input_dim

        for _ in range(num_layers):
            if init_type == 'xavier':
                std = (2 / (in_dim + hidden_dim)) ** 0.5
                W = torch.randn(hidden_dim, in_dim) * std
            elif init_type == 'kaiming':
                std = (2 / in_dim) ** 0.5
                W = torch.randn(hidden_dim, in_dim) * std
            elif init_type == 'random':
                W = torch.randn(hidden_dim, in_dim)
            else:
                raise ValueError("init_type must be 'xavier', 'kaiming' or 'random'.")

            weights.append(W)
            in_dim = hidden_dim

        x = torch.randn(input_dim)

        activation_stds = []

        for W in weights:
            x = W @ x
            x = torch.relu(x)
            activation_stds.append(round(x.std().item(), 2))

        return activation_stds
