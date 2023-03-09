from layers import *
import numpy as np
import matplotlib.pyplot as plt

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

L1 = inputLayer(X)
L2 = fullyConnectedLayer(784, 392)
L3 = ReLULayer()
L11 = dropOutLayer(0.5)
L4 = fullyConnectedLayer(392, 196)
L5 = TanhLayer()
L12 = dropOutLayer(0.5)
L6 = fullyConnectedLayer(196, 98)
L7 = TanhLayer()
L13 = dropOutLayer(0.5)
L8 = fullyConnectedLayer(98, 10)
L9 = SoftmaxLayer()
L10 = CrossEntropyLoss()

L = [L1, L2, L3, L11, L4, L5, L12, L6, L7, L13, L8, L9, L10]
Lx = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]

def accuracy(Y, Yhat):
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Yhat, axis=1))


def training(X, Y, X_test, Y_test, L, epoch = 100, early_stopping = False):
    epoch = []
    eval_train = []
    eval_test = []
    accuracy_train = []
    accuracy_test = []
    for j in range(1,100):
        ## forward Pass
        h = X
        for i in range(len(L)-1):
            h = L[i].forward(h)
        ## evaluating loss
        Loss = L[-1].eval(Y,h)
        eval_train.append(Loss)
        acc_train = accuracy(Y, h)
        accuracy_train.append(acc_train)
        epoch.append(j)
        ## backward pass
        grad = L[-1].gradient(Y,h)
        for i in range(len(L)-2,0,-1):
            newgrad = L[i].backward(grad)
            if (isinstance(L[i], fullyConnectedLayer)):
                L[i].update_weights(grad, math.pow(10, -2), j ,adam = True)
            grad = newgrad
        ## testing
        h_test = X_test
        for i in range(len(Lx)-1):
            h_test = Lx[i].forward(h_test)
        Loss = Lx[-1].eval(Y_test,h_test)
        eval_test.append(Loss)
        acc_test = accuracy(Y_test, h_test)
        accuracy_test.append(acc_test)
        print("epoch: ", j, "loss: ", Loss, "testing_accuracy: ", acc_test, "training_accuracy: ", acc_train)
        if early_stopping:
            if j > 1:
                if eval_test[-1] > eval_test[-2]:
                    break
        else:
            continue
    print("training accuracy: ", acc_train)
    print("testing accuracy: ", acc_test)
    return epoch, eval_train, eval_test, accuracy_train, accuracy_test

def kfold_cross_validation(X, Y, k, L, epoch=100, early_stopping=False):
    eval_train_kf = []
    eval_test_kf = []
    accuracy_train_kf = []
    accuracy_test_kf = []
    fold_size = len(X) // k
    for i in range(k):
        # Split the data into training and testing sets for this fold
        test_start = i * fold_size
        test_end = test_start + fold_size
        X_test = X[test_start:test_end]
        Y_test = Y[test_start:test_end]
        X_train = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        Y_train = np.concatenate((Y[:test_start], Y[test_end:]), axis=0)
        # Train the model on the training data for this fold
        epoch_kf, eval_train, eval_test, accuracy_train, accuracy_test = training(X_train, Y_train, X_test, Y_test, L, epoch=epoch, early_stopping=early_stopping)
        # Record the results for this fold
        eval_train_kf.append(eval_train)
        eval_test_kf.append(eval_test)
        accuracy_train_kf.append(accuracy_train)
        accuracy_test_kf.append(accuracy_test)
    # Calculate the average loss and accuracy across all folds
    avg_eval_train = np.mean(eval_train_kf, axis=0)
    avg_eval_test = np.mean(eval_test_kf, axis=0)
    return epoch_kf, avg_eval_train, avg_eval_test, accuracy_train_kf, accuracy_test_kf

epoch, eval_train, eval_test, accuracy_train, accuracy_test = training(X, Y, X_test, Y_test, L, epoch=100, early_stopping=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch, eval_train, label="train")
plt.plot(epoch, eval_test, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epoch, accuracy_train, label="train")
plt.plot(epoch, accuracy_test, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# epoch_kf, avg_eval_train, avg_eval_test, accuracy_train_kf, accuracy_test_kf = kfold_cross_validation(X, Y, 5, L, epoch=100, early_stopping=True)

# plt.figure(figsize=(10, 5))
# plt.plot(avg_eval_train, label="train")
# plt.plot(avg_eval_test, label="test")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.show()