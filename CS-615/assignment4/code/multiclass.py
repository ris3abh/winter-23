import numpy as np
import matplotlib.pyplot as plt

from layers import *
import math


def readCSV(filename):
    return np.genfromtxt(filename, delimiter=',')

def accuracy(Y, Yhat):
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Yhat, axis=1))


X = readCSV("/Users/rishabhsharma/Documents/GitHub/winter-23/CS-615/assignment4/data/mnist/mnist_train.csv")
Y = X[:, [0]].copy()
X = np.delete(X, 0, axis=1)   
y_abs =Y.copy()
Y = np.eye(10)[Y.astype(int).reshape(-1)]

L1 = inputLayer(X)
L2 = fullyConnectedLayer(X.shape[1], 10)
L3 = SoftmaxLayer()
L4 = CrossEntropyLoss()

L = [L1, L2, L3, L4]

h = X
threshold = math.pow(10, -10)
loss = []
epochs = []
        
L = [L1, L2, L3, L4]

h = X

eva = []
epoch = []

for i in range(1, 350):
    ## forward pass
    h = X
    for j in range(len(L)-1):
        h = L[j].forward(h)
    ## evaluating loss
    loss = L[-1].eval(Y,h)
    eva.append(loss)
    epoch.append(i)
    ## backward pass
    grad = L[-1].gradient(Y,h)
    for j in range(len(L)-2,0,-1):
        newgrad = L[j].backward(grad)
        if (isinstance(L[j], fullyConnectedLayer)):
            L[j].update_weights(grad, math.pow(10, -2), i ,adam = True)
        grad = newgrad
    if i > 2:
        if abs(eva[-2] - eva[-1]) < threshold:
            break
    print("epoch: ", i, "loss: ", loss, "accuracy: ", accuracy(Y, h))


## cross entropy loss
def cross_entropy(Y, Yhat):
    return -np.sum(Y * np.log(Yhat)) / Y.shape[0]

print("Accuracy: ", accuracy(Y, h))
print("Cross Entropy Loss: ", cross_entropy(Y, h))

## plot loss vs epochs
plt.plot(epoch, eva)
plt.xlabel('epochs')
plt.ylabel('cross entropy loss')
plt.show()

    












