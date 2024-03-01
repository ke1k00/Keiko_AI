# lesson 2
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

es = (ys-y_pre)**2

sum_e = np.sum(es)

sum_e = (1/100)*sum_e

print('sum_e',sum_e)

ws = np.arange(0,3,0.1)

es = []
for w in ws:
    y_pre = w*xs
    e = (1/100)*np.sum( (ys-y_pre)**2 )
    es.append(e)

# draw cost function 
plt.title("Cost Function", fontsize=12)
plt.xlabel("w")
plt.ylabel("e")
plt.plot(ws,es)
plt.show()

# find min pt on cost function x = -b/2a
w_min = np.sum(xs*ys)f/np.sum(xs*xs)
print("min pt on graph",str(w_min))

y_pre = w_min*xs

#check the best fit line
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs,ys)

plt.plot(xs,y_pre)

plt.show()






