# lesson 4 part 1
import dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

xs,ys = dataset.get_beans(100)

#set the graph image
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean size")
plt.ylabel("Toxicity")
plt.xlim(0,1)
plt.ylim(0,1.5)

plt.scatter(xs,ys)

w = 0.1
b = 0.1
y_pre = w*xs + b

plt.plot(xs,y_pre)

plt.show()

#my matplotlib version shows nothing
#fig = plt.figure()
#ax = Axes3D(fig)

ax = plt.axes(projection = "3d")
ax.set_zlim(0,2)

ws = np.arange(-1,2,0.1)
bs = np.arange(-2,2,0.01)

for b in bs:
    es = []
    for w in ws:
        y_pre = w*xs + b
        e = (1/100)*( np.sum((ys-y_pre)**2) )
        es.append(e)
    #plt.plot(ws,es)
    ax.plot(ws, es, b, zdir = 'y') # to make y the z-axis

plt.show()














