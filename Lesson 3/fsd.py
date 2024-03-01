# lesson 3
# fixed step descent

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
y_pre = w*xs

plt.plot(xs,y_pre)

plt.show()
        
#固定步长下降
alpha = 0.1
step = 0.01
for i in range(500):
    #k = 2aw + b
    k = 2*np.sum(xs**2)*w + np.sum(-2*xs*ys)
    k = k/100
    if k > 0:
        w = w - step
    else:
        w = w + step

    y_pre = w*xs

    #predict image of function
    plt.clf()
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(xs,y_pre)
    plt.scatter(xs,ys)
    plt.pause(0.01)



