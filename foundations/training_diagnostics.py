import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        hooks = []

        def hook_fn(module, inp, out):
            mean = out.mean().item()
            std = out.std().item()
            dead_neurons = (out<=0).all(dim=0)
            dead_fraction = (
                dead_neurons.float()
                .mean()
                .item()
            )

            res.append({
                'mean': round(mean, 4), 
                'std': round(std, 4), 
                'dead_fraction': round(dead_fraction, 4)
            })

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        model.eval()
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        return res


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        res = []
        model.train()
        model.zero_grad()
        criterion = nn.MSELoss()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                res.append({
                    'mean': round(grad.mean().item(), 4), 
                    'std': round(grad.std().item(), 4), 
                    'norm': round(torch.norm(grad).item(), 4)
                })
        return res

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction']>0.5:
                return "dead_neurons"
        for stat in gradient_stats:
            if stat["norm"]>1000:
                return "exploding_gradients"
        if gradient_stats[-1]['norm']<1e-5:
            return "vanishing_gradients"
        for stat in activation_stats:
            if stat["std"]<0.1:
                return "vanishing_gradients"
            elif stat["std"]>10.0:
                return "exploding_gradients"

        return "healthy"
