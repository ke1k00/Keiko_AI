#lesson 5
#stochastic gradient descent

#cost function
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs,ys = dataset.get_beans(100)

#set the graph image
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs,ys)

w = 0.1
b = 0.1
z = w*xs + b
a = 1/(1+ np.exp(-z))

plt.plot(xs,a)

plt.show()

for _ in range(500):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        #differentiate e wrt w&b
        z = w*x + b
        a = 1/(1+np.exp(-z) )
        e = (y-a)**2

        deda = -2*(y-a)
        dadz = a*(1-a)
        dzdw = x
        
        dedw = deda*dadz*dzdw

        dzdb = 1

        dedb = deda*dadz*dzdb

        alpha = 0.05

        w = w - alpha*dedw
        b = b - alpha*dedb

    if _%100 == 0:
        plt.clf()
        plt.scatter(xs,ys)
        z = w*xs +b
        a = 1/(1+ np.exp(-z) )

        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs,a)
        plt.pause(0.01)

