import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        batch_mu = np.mean(x, axis=0)
        batch_sig_2 = np.mean((x-batch_mu)**2, axis=0)
        
        r_m = np.array(running_mean)
        r_v = np.array(running_var)
        gamma = np.array(gamma)
        beta = np.array(beta)
        
        if training:
            r_m = (1-momentum)*r_m + momentum*batch_mu
            r_v = (1-momentum)*r_v + momentum*batch_sig_2
            x_hat = (x-batch_mu) / (np.sqrt(batch_sig_2 + eps))

        else:
            x_hat = (x-r_m) / (np.sqrt(r_v + eps))
        
        y = gamma*x_hat + beta

        return (np.round(y, 4), np.round(r_m, 4), np.round(r_v, 4))
