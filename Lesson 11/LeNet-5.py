#Lesson 11
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential #Sequential: a linear stack of layers into a Model
from keras.layers import Dense #Dense: regular deeply connected neural network layer
from keras.optimizers import SGD
import  matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# to change 2D pictures into 1D long line/array
X_train = X_train.reshape(60000, 28, 28,1)/255.0
X_test = X_test.reshape(10000, 28, 28,1)/255.0

# change numbers (e.g. 0-9) into one-hot categories
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# insert categories into CNN (convolutional neural network)
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2))) #avg pooling
model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

#training
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5000, batch_size=128)

#evaluation
loss, accuracy = model.evaluate(X_test,Y_test)
print("loss",loss)
print("accuracy",accuracy)