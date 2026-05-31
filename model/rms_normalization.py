import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)
        if len(x.shape)==1:
            rms_x = np.sqrt(np.mean(x**2) + eps)
        else:
            rms_x = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)
        x_hat = x / rms_x
        y = x_hat * gamma
        y = np.round(y, 4)
        return y.tolist()


