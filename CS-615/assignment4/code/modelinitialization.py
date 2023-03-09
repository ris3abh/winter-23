import numpy as np
import matplotlib.pyplot as plt

def plotObjectiveFunctionAndGradient(x1, w1):
    J = (1/4)*(x1*w1)**4 - (4/3)*(x1*w1)**3 + (3/2)*(x1*w1)**2
    djdw1 = w1**3 - 4*w1**2 +3*w1
    learning_rate = 0.1
    epoch = 100
    Jarray = []
    for i in range(epoch):
        w1 = w1 - learning_rate*djdw1
        Jarray.append((1/4)*(x1*w1)**4 - (4/3)*(x1*w1)**3 + (3/2)*(x1*w1)**2)
        djdw1 = w1**3 - 4*w1**2 +3*w1
    plt.plot(Jarray)
    plt.title('Objective function with w1 = ' + str(w1) + ' and J = ' + str(Jarray[-1]))
    plt.xlabel('epoch')
    plt.ylabel('J')
    plt.show()

# Plot the objective function
plotObjectiveFunctionAndGradient(1, -1)
plotObjectiveFunctionAndGradient(1, 0.2)
plotObjectiveFunctionAndGradient(1, 0.9)
plotObjectiveFunctionAndGradient(1, 4)

