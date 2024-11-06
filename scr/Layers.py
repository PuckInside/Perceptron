import numpy as np

class Layer:
    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def Backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

class ReLU(Layer):
    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        self.activation = np.maximum(0, inputs)

        return self.activation

    def Backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.derivative = np.where(self.activation > 0, 1, 0)
        input_gradient = np.multiply(gradient, self.derivative)

        return input_gradient

class Sigmoid(Layer):        
    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        self.activation = 1/(1 + np.exp(-inputs.clip(-9)))

        return self.activation
    
    def Backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.derivative = np.multiply(self.activation, (1 - self.activation))
        input_gradient = np.multiply(gradient, self.derivative)

        return input_gradient

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
            
    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = np.asmatrix(inputs)
        self.predict = np.dot(self.inputs, self.weights) + self.bias

        return np.round(self.predict)
    
    def Backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        input_gradient = np.dot(gradient, self.weights.T)
        weight_gradient = np.dot(self.inputs.transpose(), gradient)

        self.weights -= weight_gradient * learning_rate
        self.bias -= gradient * learning_rate

        return input_gradient