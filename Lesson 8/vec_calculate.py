#lesson 8
import numpy as np
import dataset
import plot_utils
m = 100
X, Y = dataset.get_beans(m)
print(X)
print()
print(Y)

plot_utils.show_scatter(X,Y)

#预测模型
# w1 = 0.1
# w2 = 0.1
W = np.array([0.1,0.1])
# b = 0.1
B = np.array([0.1])

#前向传播
def forward_propagation(X):
    # z = w1*x1s + w2*x2s + b
    Z = X.dot(W.T) + B #dot product
    # a = 1/(1+np.exp(-z))
    A = 1 / (1 + np.exp(-Z))
    return A

plot_utils.show_scatter_surface(X,Y,forward_propagation)

#用反向传播和梯度下降让曲面现出原形

for _ in range(500):
    for i in range(m):
        Xi = X[i]
        Yi = Y[i]

        A = forward_propagation(Xi) #一行一列apparently

#计算误差代价、反向传播、通过梯度下降调整参数
        #反向传播开始
        E = (Yi-A)**2
        #z = w1x1 + w2x2 + b
        #a = sigmoid(z)


        dEdA = -2*(Yi-A)
        dAdZ =A*(1-A)
        dZdW = Xi #一行两列
        dZdB = 1

        dEdW = dEdA*dAdZ*dZdW
        dEdB = dEdA*dAdZ*dZdB

        #梯度下降调整参数
        alpha = 0.01
        W = W - alpha*dEdW
        B = B - alpha*dEdB

        #反向传播结束

plot_utils.show_scatter_surface(X, Y, forward_propagation)
