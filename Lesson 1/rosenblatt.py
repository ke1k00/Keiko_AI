# lesson 1
import dataset
from matplotlib import pyplot as plt

xs,ys = dataset.get_beans(100)
print('xs')
print(xs)
print()

print('ys')
print(ys)
print()

#set the graph image
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs,ys)

w = 0.5

# rosenblatt function
for m in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        y_pre = w*x

        e = y - y_pre

        alpha = 0.05

        w = w + alpha*e*x

y_pre = w*xs

print('y_pre',y_pre)

plt.plot(xs,y_pre)

plt.show()

