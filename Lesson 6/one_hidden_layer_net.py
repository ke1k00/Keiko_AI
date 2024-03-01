# lesson 6
# stochastic gradient descent
# cost function

import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)

# set the graph image
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs, ys)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#第一层
#第一个神经元
w11_1 = np.random.rand() #From R to L: r/s betw 1st layer, 1st neuron, 1st input neuron
b1_1 = np.random.rand() #neuron position: 1st layer, 1st neuron
#第二个神经元
w12_1 = np.random.rand()
b2_1 = np.random.rand()

#第二层
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()

#forward propagation 前向传播
def forward_propagation(xs):
    #第一层第一个神经元
    z1_1 = w11_1*xs + b1_1 # find linear function
    a1_1 = sigmoid(z1_1)
    #第一层第二个神经元
    z2_1 = w12_1*xs + b2_1
    a2_1 = sigmoid(z2_1)

    #输出层
    z1_2 = w11_2*a1_1 + w21_2*a2_1 + b1_2
    a1_2 = sigmoid(z1_2)#第二层第一个神经元输出
    return a1_2,z1_2,a2_1,z2_1,a1_1,z1_1

a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propagation(xs)

plt.plot(xs,a1_2) #a1_2 final output of the neural network
plt.show()

for _ in range(5000):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        #先来一次前向传播
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(x)

#反向传播开始
        #误差代价e
        e = (y-a1_2)**2

        #求e对第二层的神经元上权重和偏置的导数
        #e对最后一个神经元的输出a1_2求导
        deda1_2 = -2*(y-a1_2)
        #a1_2对z1_2(sigmoid函数)求导
        da1_2dz1_2 = a1_2*(1-a1_2)
        #z1_2对w11_2 & w21_2求导 linear diff
        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        #chain rule 把误差e传到第二层的神经元的两个权重参数
        dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
        dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2

        #diff wrt bias term 把误差e传到第二层的神经元的偏置项
        dz1_2db1_2 = 1
        dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2



        #求e对第一层的第一个神经元上权重和偏置的导数
        dz1_2da1_1 = w11_2 #diff linear func
        da1_1dz1_1 = a1_1*(1-a1_1) #diff sigmoid func
        dz1_1dw11_1 = x #diff linear

        #求e对第一层第一个神经元和第一个输入之间的权重w11_1的导数
        #把误差e传到第一层的神经元的第一个权重参数
        dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1

        #求e对神经元的偏置项b1_1的导数
        #把误差e传到第一层的神经元的第一个偏置项
        dz1_1db1_1 = 1
        dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1db1_1



        #求e对第一层的第二个神经元上权重和偏置的导数
        dz1_2da2_1 = w21_2 #diff linear func
        da2_1dz2_1 = a2_1 * (1 - a2_1)  # diff sigmoid func
        dz2_1dw12_1 = x  # diff linear

        #求e对第一层第二个神经元和第一个输入之间的权重w12_1的导数
        #把误差e传到第一层的神经元的第二个权重参数
        dedw12_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1dw12_1

        #求e对神经元的偏置项b2_1的导数
        #把误差e传到第一层的神经元的第二个偏置项
        dz2_1db2_1 = 1
        dedb2_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1db2_1


        #梯度下降调整参数
        alpha = 0.03

        w11_1 = w11_1 - alpha * dedw11_1
        b1_1 = b1_1 - alpha * dedb1_1

        w12_1 = w12_1 - alpha * dedw12_1
        b2_1 = b2_1 - alpha * dedb2_1

        w11_2 = w11_2 - alpha * dedw11_2
        w21_2 = w21_2 - alpha*dedw21_2
        b1_2 = b1_2 - alpha * dedb1_2
#反向传播结束

    if _ % 100 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(xs)
        plt.plot(xs, a1_2)
        plt.pause(0.01)
