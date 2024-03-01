# lesson 3 -- main
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
y_pre = w*xs

plt.plot(xs,y_pre)

plt.show()

#随机梯度下降
for i in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        #e = x^2*w^2 + (-2xy)*w + y^2
        #a = x^2
        #b =-2xy
        #c = y^2
        #gradient of e, k = 2aw +b
        k = 2*(x**2)*w + (-2*x*y)
        alpha = 0.1 # to prevent excess shaking before stabilising of min pt
        w = w - alpha*k
        plt.clf() # clear previous graph
        plt.scatter(xs,ys)
        y_pre = w*xs
        # to prevent 'wobbling' dots
        plt.xlim(0,1)#fix x coords to be betw 0 and 1 
        plt.ylim(0,1.2)#fix y coords to be betw 0 and 1.2
        plt.plot(xs,y_pre)
        plt.pause(0.01) #0.01s pause between diff graphs

#reshow scatter plot and best fit line
#plt.scatter(xs,ys)
#y_pre = w*xs
#plt.plot(xs,y_pre)

#plt.show()
















    
