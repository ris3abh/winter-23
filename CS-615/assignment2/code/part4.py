import numpy as np
from layers import SquaredErrorLoss
from layers import NegativeLikelihood
from layers import CrossEntropyLoss

## instantiate each objective function

L1 = SquaredErrorLoss()
L2 = NegativeLikelihood()
L3 = CrossEntropyLoss()

## compute the loss for each objective function

Y1 = np.array([[0],[1]])
Yhat1 = np.array([[0.2],[0.3]])

print("Squared Error Loss: \n", L1.eval(Y1, Yhat1))
print("Negative Likelihood: \n", L2.eval(Y1, Yhat1))

## gradient of each objective function

print("Squared Error Loss Gradient: \n", L1.gradient(Y1, Yhat1))
print("Negative Likelihood Gradient: \n", L2.gradient(Y1, Yhat1))

Y2 = np.array([[0,1,0],[0,1,0]])
Yhat2 = np.array([[0.2,0.2,0.6],[0.2,0.7,0.1]])

print("Cross Entropy Loss: \n", L3.eval(Y2, Yhat2))

## gradient of each objective function

print("Cross Entropy Loss Gradient: \n", L3.gradient(Y2, Yhat2))