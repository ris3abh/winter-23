import numpy as np
from layers import *

## Instantiates the fully-connected layer with three inputs and two outputs

L6 = fullyConnectedLayer(3,2)

## Instantiates the weights and bias for the fully connected layer.

W = np.array([[1,2],[3, 4],[5, 6]])
b = np.array([-1, 2])

## Sets the weights and bias for the fully connected layer.

L6.setWeights(W)
L6.setBiases(b)

## H

H = np.array([[1,2,3],[4,5,6]])

## Instantiates each activation layer.

L0 = inputLayer(H)
L1 = ReLULayer()
L2 = SoftmaxLayer()
L3 = LogisticSigmoid()
L4 = TanhLayer()
L5 = LinearLayer()

## pass the data H through the forward method of each of the aforementioned layers.

Layers = [L0, L1, L2, L3, L4, L5, L6]
Layer = ["Input Layer", "ReLU Layer", "Softmax Layer", "Logistic Sigmoid Layer", "Tanh Layer", "Linear Layer", "Fully Connected Layer"]

## Forward pass with the output from the last layer

for i in range(len(Layers)):
    print("\n",Layer[i], "\n")
    H = Layers[i].forward(H)
    print(H,"\n")

   
print(H)

## calculating the gradient from the last year to the first layer

for i in range(len(Layers)-1, -1, -1):
    print("\ngradient of:\n",Layer[i], "\n")
    H = Layers[i].gradient()
    print(H,"\n")





    

