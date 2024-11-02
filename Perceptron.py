import numpy as np
from Layer import Layer

class Perceptron:
    def __init__(self, network_shape: list):
        self.layers = list()

        for i in range(len(network_shape) - 1):
            self.layers.append(Layer(network_shape[i], network_shape[i + 1]))

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
                loss += np.mean(np.power(predict - y, 2))

                output_gradient = 2 * (predict - y) / len(y)
                for layer in self.layers[::-1]:
                    output_gradient = layer.Backward(output_gradient, learning_rate)
        
            if log:
                print(f"loss - {loss / len(target)}")