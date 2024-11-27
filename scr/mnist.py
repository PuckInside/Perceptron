import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from NeuralNetwork import Perceptron
import Layers as l
import Metrics

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Размер обучающей выборки: {x_train.shape}, метки: {y_train.shape}")
print(f"Размер тестовой выборки: {x_test.shape}, метки: {y_test.shape}")

model = Perceptron([
        l.Convolutional(),
        l.Flatten((28, 28), (28*28)),
        l.Linear(28*28, 64),
        l.Sigmoid(),
        l.Linear(64, 10),
        l.Sigmoid()
])

t = np.zeros((1000, 10))
for i in range(1000):
    t[i, y_train[i]] = 1

model.Fit(x_train[:800], t[:800], 100, 0.5, True)
predict = model.Predict(x_test[0])

t = np.zeros((10000, 10))
for i in range(10000):
    t[i, y_test[i]] = 1

r2 = Metrics.R2Score(t[0], predict)
accuracy = Metrics.Accuracy(t[0], predict)

print(np.round(predict, 2))
print(t[0])

conv = l.Convolutional()
plt.imshow(conv.Forward(x_test[0]), cmap='gray')
plt.colorbar()  # Добавить шкалу значений
plt.show()

print("R2 score", r2)
print("Accuracy:", accuracy, "%")