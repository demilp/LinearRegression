import numpy as np
from enum import Enum

class LinearRegression:

    def __init__(self):
        self.weights = None

    def fit(self, X, y, learning_rate=1, epochs=1000, gradient_method='SGD'):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.ones((1, X.shape[1]))
        gradient_method = gradient_method.upper()

        if gradient_method == 'BGD':
            self.BGD(X, y, learning_rate, epochs)
        elif gradient_method == 'MBGD':
            self.MBGD(X, y, learning_rate, epochs)
        else:
            self.SGD(X, y, learning_rate, epochs)

    def predict(self, X):
        return X.dot(self.weights.T)

    def gradients(self, X, y):
        grads = np.zeros((X.shape[1], 1))
        p = self.predict(X)
        dif = p - y
        for j in range(self.weights.shape[1]):
            f = X[:, j][np.newaxis].T
            grads[j] = dif.T.dot(f) * 2/X.shape[0]
        return grads.T

    def learning_schedule(t, t0, t1):
        return t0/(t+t1)

    def BGD(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            self.weights = self.weights - learning_rate * self.gradients(X, y)

    def MBGD(self, X, y, learning_rate, epochs):
        m = X.shape[0]
        size = int(m*0.1)+1
        for e in range(epochs):
            for i in range(m):
                index = np.random.randint(m-size)
                self.weights = self.weights - LinearRegression.learning_schedule(epochs * m + i, epochs / 10, epochs) * learning_rate * self.gradients(X[index:index + size], y[index:index + size])

    def SGD(self, X, y, learning_rate, epochs):
        m = X.shape[0]
        for e in range(epochs):
            for i in range(m):
                index = np.random.randint(m)
                self.weights = self.weights - LinearRegression.learning_schedule(epochs * m + i, epochs / 10, epochs) * learning_rate * self.gradients(X[index:index + 1], y[index:index + 1])

    def rmse(self, X, y):
        return np.sqrt(((self.predict(X)-y)**2).mean)


