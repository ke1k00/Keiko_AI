#Lesson 10 -- detect image
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential #Sequential: a linear stack of layers into a Model
from keras.layers import Dense #Dense: regular deeply connected neural network layer
from keras.optimizers import SGD
import  matplotlib.pyplot as plt
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('X_train shape',str(X_train.shape))
print('Y_train shape',str(Y_train.shape))
print('X_test shape',str(X_test.shape))
print('Y_test shape',str(X_test.shape))

print(Y_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()

# to change 2D pictures into 1D long line/array
X_train = X_train.reshape(60000, 784)/255.0
X_test = X_test.reshape(10000, 784)/255.0

# change numbers (e.g. 0-9) into one-hot categories
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)


model = Sequential()
# dense = Dense(units = 1, activation = "sigmoid", input_dim = 1)
#units: 某层有几个神经元, activation: 激活函数类型, 数据特征维度/no. of inputs to the neuron: 1
# model.add(dense)

model.add(Dense(units = 256, activation = "relu", input_dim = 784))
model.add(Dense(units = 256, activation = "relu"))
model.add(Dense(units = 256, activation = "relu"))
model.add(Dense(units = 10, activation = "softmax"))


model.compile(loss= 'categorical_crossentropy', optimizer= SGD(lr = 0.05), metrics= ['accuracy'])
#loss 代价函数: 均方误差, optimizer 优化器: 随机梯度下降算法, metrics 评估标准: 准确度
#lr: learning rate 学习率

model.fit(X_train, Y_train, epochs= 5000, batch_size= 128)
# epochs: no. of iteration (aft all testing done -- 1 iteration) over the entire x and y data
# batch_size: no. of data to be taken out for testing

loss, accuracy = model.evaluate(X_test,Y_test)
print("loss",loss)
print("accuracy",accuracy)