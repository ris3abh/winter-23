import numpy as np
import matplotlib.pyplot as plt

# define the function
x1 = 1
x2 = 1
w1 = 0
w2 = 0

J = (x1*w1 - 5*x2*w2 - 2)**2

# define the partial derivatives
dJdw1 = 2*(x1*w1 - 5*x2*w2 - 2)*x1
dJdw2 = 2*(x1*w1 - 5*x2*w2 - 2)*(-5*x2)

learning_rate = 0.01
epoch = 100

# gradient descent
w1Array = []
w2Array = []
JArray = []
for i in range(epoch):
    w1Array.append(w1)
    w2Array.append(w2)
    JArray.append(J)
    w1 = w1 - learning_rate*dJdw1
    w2 = w2 - learning_rate*dJdw2

    J = (x1*w1 - 5*x2*w2 - 2)**2
    dJdw1 = 2*(x1*w1 - 5*x2*w2 - 2)*x1
    dJdw2 = 2*(x1*w1 - 5*x2*w2 - 2)*(-5*x2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(w1Array, w2Array, JArray)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('J')
plt.show()