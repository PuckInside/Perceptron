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
        l.Linear(28*28, 100),
        l.Sigmoid(),
        l.Linear(100, 36),
        l.Sigmoid(),
        l.Linear(36, 10),
        l.Sigmoid()
])

train = np.zeros((y_train.shape[0], 10))
for i in range(y_train.shape[0]):
    train[i, y_train[i]] = 1

model.Fit(x_train[:4200], train[:4200], 40, 0.288, True)

test = np.zeros((y_test.shape[0], 10))
for i in range(y_test.shape[0]):
    test[i, y_test[i]] = 1

correct = 0

for i in range(100):
    predict = model.Predict(x_test[i])
    r2 = Metrics.R2Score(test[i], predict)

    if r2 > 0.3:
        correct += 1

    print(np.round(predict, 2))
    print(test[i])

    print("R2 score", r2)
    print("***")


print(correct)
conv = l.Convolutional()
plt.imshow(conv.Forward(x_test[0]), cmap='gray')
plt.colorbar()  # Добавить шкалу значений
plt.show()
