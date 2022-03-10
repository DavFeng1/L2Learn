import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
np.seterr(all='raise')

testDataFrame = pd.read_csv("./datasets/mnist/mnistTest.csv")
trainDataFrame = pd.read_csv("./datasets/mnist/mnistTrain.csv")
testData = np.array(testDataFrame)
trainData = np.array(trainDataFrame)

# Normalize data
epsilon = 0.1
Y = np.array([[1 if label == i else 0 for i in range(10)] 
  for label in trainData[:,0]])

X = trainData[:,1:] # The input vector
X = X / 255
n = 784 # Dimension of the input space (28 * 28 = 784 pixels)
m = 16 # Dimension of hidden layer ( 4 nodes )
l = 10 # Dimension of output ( 10 possible digits to classify )
k = 60000 # Size of training data

# Define functions
def sigmoid(x):
  return 1 / ( 1 + np.exp(-x))

def sigmoidPrime(x):
  return np.exp(-x) / ( 1 + np.exp(-x)) ** 2

def ReLU(Z):
  return np.maximum(Z, 0)

def ReluPrime(Z):
  return 1 * ( Z > 0 )

def softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A

def error(P, L):
  if (P.shape != L.shape):
    raise ValueError("Lengths must be the same")

  return 0.5 *  np.sum([(P[i] - L[i])**2 for i in range(l)])


def initializeParameters():
  # Initialize weights for the input and the hidden layer
  # Between -0.5 and 0.5 since we are using ReLU as activation function
  W0 = np.random.rand(n, m) - 0.5
  W1 = np.random.rand(m, l) - 0.5
  b0 = np.random.rand(m) - 0.5
  b1 = np.random.rand(l) - 0.5

  return W0, b0, W1, b1

def forwardProp(W0, W1, b0, b1, X):
  A0 = W0.T.dot(X) + b0
  Z0 = ReLU(A0)
  Z1 = W1.T.dot(Z0) + b1
  P = softmax(Z1)

  if not math.isclose(np.sum(P), 1.0):
    raise ValueError("forwardProp: P does not sum to 1")

  return A0, P

def backProp(W1, P, X, Y, A0):

  Z = (P - Y) * P * (1 + P)

  dW0 = W1.dot(Z) * np.outer(X, ReluPrime(A0))
  db0 = ReluPrime(A0) * W1.dot(Z)
  dW1 = np.outer(Z, ReLU(A0))
  db1 =  Z

  return dW0, db0, dW1, db1


def getPredictions(P):
    return np.argmax(P, 0)

def getAccuracy(p, Y):
  return np.sum(p == Y) / Y.size

def updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon):
  # W0_prev = W0
  # W1_prev = W1
  W0 = W0 - epsilon * dW0
  b0 = b0 - epsilon * db0
  W1 = W1 - epsilon * dW1.T
  b1 = b1 - epsilon * db1

  # print("========== W0 Diff", np.sum(W0 - W0_prev))
  # print("========== W1 Diff", np.sum(W1 - W1_prev))

  return W0, b0, W1, b1

def gradientDescent(X, Y, epsilon, iterations):
  
  W0, b0, W1, b1 = initializeParameters()
  errors = []
  for i in range(iterations):
    A0, P = forwardProp(W0, W1, b0, b1, X[i])
    dW0, db0, dW1, db1 = backProp(W1, P, X[i], Y[i], A0)
    W0, b0, W1, b1 = updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon)

    if i % 100 == 0:
      print("Iteration: ===================", i)
      currError = error(P, Y[i])
      errors.append(currError)
      print("Error: ", currError)

  plt.plot(errors)
  plt.ylabel("Errors")
  plt.savefig('errors.png')
  return W0, b0, W1, b1


W0, b0, W1, b1 = gradientDescent( X, Y, epsilon, 60000)
# %%
