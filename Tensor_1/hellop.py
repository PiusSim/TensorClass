import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
키로부터 몸무게를 예측하고 싶다
x : 키
y : 몸무게
m :  5
'''
키 = np.array([150, 160, 170, 180, 185])
몸무게 = np.array([50, 55, 60, 68, 72])
키 = np.reshape(키, [-1, 1])
키_test = [165, 190]

model = LinearRegression().fit(키, 몸무게)
print('a', model.coef_, 'b', model.intercept_)
plt.plot(키, 몸무게)
x = np.arange(140, 200, 10)
y = model.coef_ * x + model.intercept_

plt.plot(x, y)
plt.scatter(키, 몸무게)
plt.xlabel('key')
plt.ylabel('kg')
plt.title('height/weight')
plt.show()




