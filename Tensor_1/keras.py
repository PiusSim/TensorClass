import numpy as np
import keras


m = 5
x = np.array([150, 160, 170, 180, 185])/200
y = np.array([50, 55, 60, 68, 72])/100
#키 = np.reshape(키, [-1, 1])
#키_test = [165, 190]

model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(loss='mse', optimizer='sgd')




