## import layers from the layers.py file

from layers import *

X = np.array([[1,2,3,4],[5,6,7,8]])

## call the input layer

# inputLayer = inputLayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = inputLayer.forward(X)
# print("output from input layer is: \n",dataOut, "\n")

# ## call the linear layer

# linearLayer = LinearLayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = linearLayer.forward(X)
# print("output from linear layer is: \n",dataOut, "\n")


# ## call the logistic Layer

# logisticLayer = logisticSigmoidLayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = logisticLayer.forward(X)
# print("output from logistic layer is: \n",dataOut, "\n")


# ## call the relu layer

# reluLayer = RelULayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = reluLayer.forward(X)
# print("output from relu layer is: \n",dataOut, "\n")


# ## call the softmax layer

# softmaxLayer = SoftMaxLayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = softmaxLayer.forward(X)
# print("output from softmax layer is: \n",dataOut, "\n")


# ## call the tanh layer

# tanhLayer = tanHLayer(X)
# print("input layer is: \n",X, "\n")

# dataOut = tanhLayer.forward(X)
# print("output from tanh layer is: \n",dataOut, "\n")


# ## call the fully connected layer

# fullyConnectedLayer = fullyConnectedLayer(4, 2)
# print("input layer is: \n",X, "\n")

# dataOut = fullyConnectedLayer.forward(X)
# print("output from fully connected layer is: \n",dataOut, "\n")


# ## call the fully connected layer 2 with output size 2

weights = np.array([[1,2],[2,0],[1,1],[-1,4]])
bias = np.array([[1,2]])

# fullyConnectedLayer2 = fullyConnectedLayer2(4, 2, weights, bias)
# print("input layer is: \n",X, "\n")

# dataOut = fullyConnectedLayer2.forward(X)
# print("output from fully connected layer 2 is: \n",dataOut, "\n")


## calling the input layer feeding its output to the fully connected layer with 2 outputs and then feeding its output to the sigmoid layer

print("\n")
inputLayer = inputLayer(X)
print("input layer is: \n",X, "\n")

dataOut = inputLayer.forward(X)
print("output from input layer is: \n",dataOut, "\n")

fullyConnectedLayer2 = fullyConnectedLayer2(4, 2, weights, bias)

dataOut1 = fullyConnectedLayer2.forward(dataOut)
print("output from fully connected layer 2 is: \n",dataOut1, "\n")

logisticLayer = logisticSigmoidLayer(dataOut1)

dataOut2 = logisticLayer.forward(dataOut1)
print("output from logistic layer is: \n",dataOut2, "\n")




