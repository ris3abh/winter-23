import numpy as np
import matplotlib.pyplot as plt
x1 = 1
w1 = np.arange(-2, 5, 0.1)

Jval = []
for i in range(len(w1)):
    Jval.append((1/4)*(x1*w1[i])**4 - (4/3)*(x1*w1[i])**3 + (3/2)*(x1*w1[i])**2)

# Plot the objective function
plt.plot(w1, Jval)
plt.xlabel('w1')
plt.ylabel('J')
plt.title('Objective function')
plt.show()