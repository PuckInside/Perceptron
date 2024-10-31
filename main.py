import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Perceptron import Perceptron

def R2Score(y_true: np.ndarray, y_pred: np.ndarray):
    y_mean = y_true.mean()
    ss_res = np.sum(np.power((y_true - y_pred), 2))
    ss_tot = np.sum(np.power(y_true - y_mean, 2))

    r2 = 1 - (ss_res / ss_tot)
    return r2

def Accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    return accuracy

data = pd.read_csv("C:/Users/sagin/.cache/kagglehub/datasets/uciml/iris/versions/2/Iris.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = np.asmatrix(data.drop(columns=['Species', 'Id']))
y = np.asmatrix(pd.get_dummies(data['Species']).astype(float).values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Perceptron([4, 128, 64, 3])
model.Fit(X_train, y_train, 10000, 0.1)

predict = model.Predict(X_test)
r2 = R2Score(y_test, predict)
accuracy = Accuracy(y_test, predict)

print(predict[0])
print(y_test[0])
print("R2 score", np.round(r2, 3))
print("Accuracy:", np.round(accuracy, 3), "%")