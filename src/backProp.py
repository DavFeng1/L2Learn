import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from activations import ReLU, ReluPrime, softmax
from cost import meanSquareError

np.seterr(all='raise')

testDataFrame = pd.read_csv("./datasets/mnist/mnistTest.csv")
trainDataFrame = pd.read_csv("./datasets/mnist/mnistTrain.csv")
testData = np.array(testDataFrame)
trainData = np.array(trainDataFrame)

# Normalize data
epsilon = 0.01
Y = np.array([[1 if label == i else 0 for i in range(10)] 
  for label in trainData[:,0]])

X = trainData[:,1:] # The input vector
X = X / 255
n = 784 # Dimension of the input space (28 * 28 = 784 pixels)
m = 16 # Dimension of hidden layer ( 4 nodes )
l = 10 # Dimension of output ( 10 possible digits to classify )
k = 60000 # Size of training data

# Plot 10 numbers
# fig, axes = plt.subplots(2,5, figsize=(12,5))
# axes = axes.flatten()
# for i in range(10):
#     axes[i].imshow(X[i].reshape(28,28), cmap='gray')
# plt.savefig("numbers")

def initializeParameters():
  W0 = (1/n) * (np.random.rand(m, n) - 0.5)
  W1 = (1/m) * (np.random.rand(l, m) - 0.5)
  b0 = (1/n) * (np.random.rand(m) - 0.5)
  b1 =  (1/m) * (np.random.rand(l) - 0.5)

  return W0, b0, W1, b1

def forwardProp(W0, W1, b0, b1, X_i):
  Z0 = W0.dot(X_i) + b0
  A0 = ReLU(Z0)
  Z1 = W1.dot(A0) + b1
  P = softmax(Z1)

  if not math.isclose(np.sum(P), 1.0):
    raise ValueError("forwardProp: P does not sum to 1")

  return Z0, P

def backProp(W1, P, X, Y, Z0):

  PY = (P - Y) * P * (1 + P)

  dW0 = np.outer(ReluPrime(Z0).dot(W1.T.dot(PY)), X)
  db0 = ReluPrime(Z0) * W1.T.dot(PY)
  dW1 = np.outer(PY, ReLU(Z0))
  db1 = PY

  return dW0, db0, dW1, db1


def getPredictions(P):
    return np.argmax(P, 0)

def getAccuracy(p, Y):
  return np.sum(p == Y) / Y.size

def updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon):
  W0 = W0 - epsilon * dW0
  b0 = b0 - epsilon * db0
  W1 = W1 - epsilon * dW1
  b1 = b1 - epsilon * db1

  return W0, b0, W1, b1

def singlePass():
  pass

def gradientDescent(X, Y, epsilon, epochs):
  
  W0, b0, W1, b1 = initializeParameters()
  w1_average = []
  w0_average = []
  b0_average = []
  b1_average = []
  error_average = []

  for i in range(epochs):
    for j in range(k):
      Z0, P = forwardProp(W0, W1, b0, b1, X[j])
      dW0, db0, dW1, db1 = backProp(W1, P, X[j], Y[j], Z0)
      W0, b0, W1, b1 = updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon)

      if j % 1000 == 0:
        w1_average.append(np.sum(W1) / len(W1))
        currError = meanSquareError(P, Y[i])
        error_average.append(currError)

        if j % 7500 == 0:
          print("================ Epoch: {}, iteration: {} ===================".format(i, j))
          print("Error: ", currError)

  plt.plot(error_average, label="Error")
  plt.plot(w1_average, label="W1 Average")
  plt.savefig("trainingCurve.png")
  return W0, b0, W1, b1


W0, b0, W1, b1 = gradientDescent( X, Y, epsilon, 5)
# %%
