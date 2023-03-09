import numpy as np
import matplotlib.pyplot as plt

w1 = 0.2
learning_rate = 5
rho1 = 0.9
rho2 = 0.999
delta = 1e-8

## applying adam
def adam(w1, learning_rate, rho1, rho2, delta):
    epoch = 100
    Jarray = []
    s, r = 0, 0
    for i in range(1,epoch):
        Jarray.append((1/4)*(w1**4) - (4/3)*(w1**3) + (3/2)*(w1**2))
        djdw1 = (w1**3) - (4* (w1**2)) + (3 * w1)
        s = (rho1 * s) + (1-rho1) * djdw1
        r = (rho2 * r) + (1-rho2) * (djdw1**2)
        s_hat = s/(1-(rho1**i))
        r_hat = r/(1-(rho2**i))
        w1 = w1 - (learning_rate * ((s_hat)/(np.sqrt(r_hat) + delta)))

    plt.plot(Jarray)
    plt.show()
    plt.title('Objective function versus epoch at learning rate = ' + str(learning_rate))

# Plot the objective function
adam(0.2, 5, 0.9, 0.999, 1e-8)
