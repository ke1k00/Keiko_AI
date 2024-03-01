#Lesson 9 -- use of keras model
import dataset
import numpy as np
import plot_utils
from keras.models import Sequential #Sequential: a linear stack of layers into a Model
from keras.layers import Dense #Dense: regular deeply connected neural network layer
from keras.optimizers import SGD

m = 100
X,Y = dataset.get_beans(m)
plot_utils.show_scatter(X,Y)

model = Sequential()
# dense = Dense(units = 1, activation = "sigmoid", input_dim = 1)
#units: 某层有几个神经元, activation: 激活函数类型, 数据特征维度/no. of inputs to the neuron: 1
# model.add(dense)

model.add(Dense(units = 8, activation = "relu", input_dim = 2))
model.add(Dense(units = 8, activation = "relu"))
model.add(Dense(units = 8, activation = "relu"))
model.add(Dense(units = 1, activation = "sigmoid"))

model.compile(loss= 'mean_squared_error', optimizer= SGD(lr = 0.05), metrics= ['accuracy'])
#loss 代价函数: 均方误差, optimizer 优化器: 随机梯度下降算法, metrics 评估标准: 准确度
#lr: learning rate 学习率

model.fit(X, Y, epochs= 5000, batch_size= 10)
# epochs: no. of iteration (aft all testing done -- 1 iteration) over the entire x and y data
# batch_size: no. of data to be taken out for testing

pres = model.predict(X)

plot_utils.show_scatter_surface(X, Y, model)

print(model.get_weights())