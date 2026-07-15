import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        activations = []
        hooks = []
        
        # Hook函数: 保存ReLU输出
        def hook(module, input, output):
            activations.append(output)

        # 给所有ReLU注册hook
        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook))

        # Forward
        with torch.no_grad():
            model(x)

        # 删除hook
        for h in hooks:
            h.remove()

        dead_fractions = []

        for output in activations:
            # output shape: (batch_size, hidden_dim, ...)
            # neuron对应除batch之外的每一个位置
            batch = output.shape[0]
            output = output.reshape(batch, -1)

            # 对batch维判断是否始终为0
            dead = (output==0).all(dim=0)
            fraction = dead.float().mean().item()

            dead_fractions.append(round(fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(f>0.5 for f in dead_fractions):
            return "use_leaky_relu"

        if len(dead_fractions)>0 and dead_fractions[0]>0.3:
            return "reinitialize"

        increasing = all(
            dead_fractions[i]>dead_fractions[i-1] 
            for i in range(1, len(dead_fractions))
        )

        if (
            increasing
            and len(dead_fractions)>0
            and dead_fractions[-1]>0.1
        ):
            return "reduce_learning_rate"

        if max(dead_fractions, default=0)<0.1:
            return "healthy"

        return "healthy"
