import numpy as np

from Layers import Layer
from Losses import MSE

class Perceptron:
    def __init__(self, network_shape: list):
        self.loss = MSE()
        self.layers = network_shape

    def Predict(self, inputs: list) -> list:
        predict = inputs
        for layer in self.layers:
            predict = layer.Forward(predict)

        return predict

    def Fit(self, inputs: np.ndarray, target: np.ndarray, epochs: int, learning_rate: float = 0.1, log: bool = False):
        for e in range(epochs):
            loss = 0
            for x, y in zip(inputs, target):
                predict = self.Predict(x)
                loss += self.loss.CalculateLoss(y, predict)

                output_gradient = self.loss.CalculateDerivative(y, predict)
                for layer in self.layers[::-1]:
                    output_gradient = layer.Backward(output_gradient, learning_rate)
        
            if log:
                print(f"loss - {loss / len(target)}")