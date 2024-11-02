import numpy as np

def R2Score(y_true: np.ndarray, y_pred: np.ndarray, round = 3):
    y_mean = y_true.mean()
    ss_res = np.sum(np.power((y_true - y_pred), 2))
    ss_tot = np.sum(np.power(y_true - y_mean, 2))

    r2 = 1 - (ss_res / ss_tot)
    return np.round(r2, round)

def Accuracy(y_true: np.ndarray, y_pred: np.ndarray, round = 3):
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    return np.round(accuracy, round)