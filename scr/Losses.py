import numpy as np

class Loss:
    def CalculateLoss(self, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
        pass

    def CalculateDerivative(self, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
        pass

class MSE(Loss):
    def CalculateLoss(self, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
        self.loss = np.mean(np.power(predict - target, 2))

        return self.loss

    def CalculateDerivative(self, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
        self.derivative = 2 * (predict - target) / len(target)

        return self.derivative
