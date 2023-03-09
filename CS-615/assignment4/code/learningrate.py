import numpy as np
import matplotlib.pyplot as plt

def plotObjectiveFunctionAndGradient(x1, w1, learning_rate):
    J = (1/4)*(w1 ** 4) - (4/3)*(w1 ** 3) + (3/2)*(w1 ** 2)
    djdw1 = w1**3 - 4*w1**2 +3*w1
    epoch = 100
    Jarray = []
    for i in range(epoch):
        try:
            j = (1/4)*(w1 ** 4) - (4/3)*(w1 ** 3) + (3/2)*(w1 ** 2)
            djdw1 = (w1 ** 3) + (-4 *(w1 ** 2)) + (3 * w1)
        except:
            break
        Jarray.append(j)
        w1 = w1 - learning_rate*djdw1
    plt.plot(Jarray)
    plt.title("final value of w1 and J are: " + str(w1) + " and " + str(Jarray[-1]))
    plt.xlabel('epoch')
    plt.ylabel('J')
    plt.show()
    

# Plot the objective function
plotObjectiveFunctionAndGradient(1, 0.2, 0.001)
plotObjectiveFunctionAndGradient(1, 0.2, 0.01)
plotObjectiveFunctionAndGradient(1, 0.2, 1)
plotObjectiveFunctionAndGradient(1, 0.2, 5)