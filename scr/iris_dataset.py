import pandas as pd
from sklearn.model_selection import train_test_split

from NeuralNetwork import Perceptron
import Layers as layer
import Metrics

data = pd.read_csv("C:/Users/sagin/.cache/kagglehub/datasets/uciml/iris/versions/2/Iris.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.drop(columns=['Species', 'Id']).values
y = pd.get_dummies(data['Species']).astype(float).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Perceptron([
        layer.Linear(4, 20),
        layer.ReLU(),
        layer.Linear(20, 3)
        ])

model.Fit(X_train, y_train, 1000, 0.001, True)
predict = model.Predict(X_test)

r2 = Metrics.R2Score(y_test, predict)
accuracy = Metrics.Accuracy(y_test, predict)

print("R2 score", r2)
print("Accuracy:", accuracy, "%")