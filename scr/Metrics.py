import numpy as np

def R2Score(target: np.ndarray, y_pred: np.ndarray, round = 3) -> float:
    y_mean = target.mean()
    ss_res = np.sum(np.power((target - y_pred), 2))
    ss_tot = np.sum(np.power(target - y_mean, 2))

    r2 = 1 - (ss_res / ss_tot)
    return np.round(r2, round)

def Accuracy(target: np.ndarray, predict: np.ndarray, round = 3) -> float:
    predict = (predict > 0.6).astype(int)
    accuracy = np.mean(np.array(target) == np.array(predict)) * 100
    return np.round(accuracy, round)
 