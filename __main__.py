import numpy as np
from LinearRegression import LinearRegression

amp = 5;
X = np.random.rand(100, 1)
y = 4 + 8 * X + np.random.rand(100, 1)*amp-0.5*amp
lr = LinearRegression()
lr.fit(X, y, gradient_method='MBGD')
print(lr.weights)
print(lr.rmse(X, y))
lr.fit(X, y, gradient_method='BGD', learning_rate=0.1)
print(lr.weights)
print(lr.rmse(X, y))
lr.fit(X, y, gradient_method='SGD')
print(lr.weights)
print(lr.rmse(X, y))
