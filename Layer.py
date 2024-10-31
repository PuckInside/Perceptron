import numpy as np
from Sigmoid import Sigmoid

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        self.activation_function = Sigmoid()
            
    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.predict = np.dot(self.inputs, self.weights) + self.bias
        self.activation_result = self.activation_function.Activate(self.predict)

        return np.round(self.activation_result, 3)
    
    def Backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        gradient = np.multiply(gradient, self.activation_function.ActivatationDerivative())
        input_gradient = np.dot(gradient, self.weights.T)
        weight_gradient = np.dot(self.inputs.transpose(), gradient)

        self.weights -= weight_gradient * learning_rate
        self.bias -= gradient * learning_rate

        return input_gradient