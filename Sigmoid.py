import numpy as np

class Sigmoid:
    def Activate(self, predict: np.ndarray) -> np.ndarray:
        self.activation = 1/(1 + np.exp(-predict.clip(-9)))
        return self.activation

    def ActivatationDerivative(self) -> np.ndarray:
        self.derivative = np.multiply(self.activation, (1 - self.activation))
        return self.derivative
    