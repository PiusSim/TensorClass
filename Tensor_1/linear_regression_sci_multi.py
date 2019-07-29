import sklearn as tf
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor

data = np.array([
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196],
    [73., 66., 70., 142.]
], dtype=np.float32) / 200

x = data[:, :-1]
y = data[:, [-1]]

model = SGDRegressor().fit(x, y)
print(model.coef_, model.intercept_)
