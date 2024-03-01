#lesson 4 part 2
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
y_pre = w*xs + b

plt.plot(xs,y_pre)

plt.show()

#随机梯度下降 -- to train till min. pt

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        #e = (y - (wx+b) )^2 = x^2w^2 + (2xb-2xy)w + (y^2+b^2-2yb)
        #e (dw), dw = 2x^2w + (2xb - 2xy)
        dw = 2*x**2*w + 2*x*b - 2*x*y
        #e = (y - (wx+b) )^2 = b^2 + (2xw - 2y)b + (x^2w^2 + y^2 - 2xyw)
        #e (db), db = 2b + (2xw - 2y)
        db = 2*b + 2*x*w - 2*y
        
        alpha = 0.01 #to prevent excess shaking before stabilising of min pt
        w = w - alpha*dw
        b = b - alpha*db


    plt.clf() #clear previous graph
    plt.scatter(xs,ys)
    y_pre = w*xs + b
    #to prevent 'wobbling' dots
    plt.xlim(0,1)#fix x coords to be betw 0 and 1 
    plt.ylim(0,1.2)#fix y coords to be betw 0 and 1.2
    plt.plot(xs,y_pre)
    plt.pause(0.01)#0.01s pause between diff graphs

#reshow scatter plot and best fit line
#plt.scatter(xs,ys)
#y_pre = w*xs
#plt.plot(xs,y_pre)

#plt.show()
