import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        assert len(weights)==len(biases)
        h = x
        for i in range(len(weights)-1):
            h = h.dot(weights[i])+biases[i]
            h = np.maximum(h, 0)
        h_final = h.dot(weights[-1])+biases[-1]
        return np.round(h_final, 5)
