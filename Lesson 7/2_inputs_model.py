#lesson 7
import numpy as np
import dataset
import plot_utils
m = 100
xs, ys = dataset.get_beans(m)
print(xs)
print()
print(ys)

plot_utils.show_scatter(xs,ys)

#预测模型
w1 = 0.1
w2 = 0.1
b = 0.1

#前向传播
#z = w1x1 + w2x2 + b
x1s = xs[:,0] # array slicing --> grp all index[0] elemts of xs to form a new array
x2s = xs[:,1]

def forward_propagation(x1s,x2s):
    z = w1*x1s + w2*x2s + b
    a = 1/(1+np.exp(-z))
    return a

plot_utils.show_scatter_surface(xs,ys,forward_propagation)

#用反向传播和梯度下降让曲面现出原形

for _ in range(500):
    for i in range(m):
        x = xs[i] #xs: 2d arr
        y = ys[i]
        x1 = x[0]
        x2 = x[1]

        a = forward_propagation(x1,x2)

#计算误差代价、反向传播、通过梯度下降调整参数
        #反向传播开始
        e = (y-a)**2
        #z = w1x1 + w2x2 + b
        #a = sigmoid(z)

        deda = -2*(y-a)
        dadz = a*(1-a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        dedw1 = deda*dadz*dzdw1
        dedw2 = deda*dadz*dzdw2
        dedb = deda*dadz*dzdb

        #梯度下降调整参数
        alpha = 0.01
        w1 = w1 - alpha*dedw1
        w2 = w2 - alpha*dedw2
        b = b - alpha*dedb

        #反向传播结束

plot_utils.show_scatter_surface(xs, ys, forward_propagation)
