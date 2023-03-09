import numpy as np
import matplotlib.pyplot as plt

from layers import *
import math


def readCSV(filename):
    return np.genfromtxt(filename, delimiter=',')

def accuracy(Y, Yhat):
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Yhat, axis=1))

X = readCSV("../data/mnist/mnist_train_100.csv")
Y = X[:, [0]].copy()
X = np.delete(X, 0, axis=1)   
y_abs =Y.copy()
Y = np.eye(10)[Y.astype(int).reshape(-1)]

X_test = readCSV("../data/mnist/mnist_valid_10.csv")
Y_test = X_test[:, [0]].copy()
X_test = np.delete(X_test, 0, axis=1)
y_abs_test = Y_test.copy()
Y_test = np.eye(10)[Y_test.astype(int).reshape(-1)]

def training(X, Y, X_test, Y_test, L, LX, epoch = 100, early_stopping = False):
    epochs = []
    eval_train = []
    eval_test = []
    accuracy_train = []
    accuracy_test = []
    for j in range(1,epoch+1):
        ## forward Pass
        h = X
        for i in range(len(L)-1):
            h = L[i].forward(h)
        ## evaluating loss
        Loss_train = L[-1].eval(Y,h)
        eval_train.append(Loss_train)
        acc_train = accuracy(Y, h)
        accuracy_train.append(acc_train)
        epochs.append(j)
        ## backward pass
        grad = L[-1].gradient(Y,h)
        for i in range(len(L)-2,0,-1):
            newgrad = L[i].backward(grad)
            if (isinstance(L[i], fullyConnectedLayer)):
                L[i].update_weights(grad, math.pow(10, -3), j ,adam = True)
            grad = newgrad
        ## testing
        h_test = X_test
        for i in range(len(LX)-1):
            h_test = LX[i].forward(h_test)
        Loss_test = LX[-1].eval(Y_test,h_test)
        eval_test.append(Loss_test)
        acc_test = accuracy(Y_test, h_test)
        accuracy_test.append(acc_test)
        print("epoch: ", j, "loss: ", Loss_train, "testing_accuracy: ", acc_test, "training_accuracy: ", acc_train)
        if early_stopping:
            if j > 1:
                if eval_test[-1] > eval_test[-2]:
                    break
        else:
            continue
    print("training accuracy: ", acc_train)
    print("testing accuracy: ", acc_test)
    return epochs, eval_train, eval_test, accuracy_train, accuracy_test

def plot(epoch, eval_train, eval_test, accuracy_train, accuracy_test):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, eval_train, label = "training")
    plt.plot(epoch, eval_test, label = "testing")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epoch, accuracy_train, label = "training")
    plt.plot(epoch, accuracy_test, label = "testing")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def trainingAutoEncoder(L, input, output, epochs = 100):
    epoch = []
    eval_train = []
    for j in range(1,epochs):
        ## forward pass
        h = input
        for i in range(len(L)-1):
            h = L[i].forward(h)
        ## evaluating loss
        error = L[-1].eval(output,h)
        eval_train.append(error)
        epoch.append(j)
        ## backward pass
        grad = L[-1].gradient(output,h)
        for i in range(len(L)-2,0,-1):
            newgrad = L[i].backward(grad)
            if (isinstance(L[i], fullyConnectedLayer)):
                L[i].update_weights(grad, math.pow(10, -3), j ,adam = True)
            grad = newgrad
        print("epoch: ", j, "loss: ", error)
    yhat = input
    for i in range(len(L)-2):
        yhat = L[i].forward(yhat)
    return yhat, L[-3]

Lfirst = [inputLayer(X), fullyConnectedLayer(784,256), fullyConnectedLayer(256, 784), SquaredErrorLoss()]
yhat1, FC1 = trainingAutoEncoder(Lfirst, X, X, epochs = 500)
Lsecond = [inputLayer(yhat1), fullyConnectedLayer(256, 128), fullyConnectedLayer(128, 256), SquaredErrorLoss()]
yhat2, FC2,  = trainingAutoEncoder(Lsecond, yhat1, yhat1, epochs = 500)
Lthird = [inputLayer(yhat2), fullyConnectedLayer(128, 10), SoftmaxLayer(), CrossEntropyLoss()]
yhat3, FC3,  = trainingAutoEncoder(Lthird, yhat2, Y, epochs = 500)

L1 = inputLayer(X)
L2 = FC1
L3 = ReLULayer()
Ln = dropOutLayer(0.2)
L4 = FC2
L5 = ReLULayer()
Ln1 = dropOutLayer(0.2)
L6 = FC3
L7 = SoftmaxLayer()
L8 = CrossEntropyLoss()

L = [L1, L2, L3, L4, L5, L6, L7, L8]
LX = [L1, L2, L3, L4, L5, L6, L7, L8]


epoch, eval_train, eval_test, accuracy_train, accuracy_test = training(X, Y, X_test, Y_test, L, LX,epoch = 500, early_stopping = True)

plot(epoch, eval_train, eval_test, accuracy_train, accuracy_test)

## without early stopping
## accuracy train: 0.931
## accuracy test: 0.81

## with early stopping
## accuracy train: 0.836
## accuracy test: 0.77