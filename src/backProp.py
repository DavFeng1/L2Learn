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
epsilon = 0.001
Y = np.array([[1 if label == i else 0 for i in range(10)] 
  for label in trainData[:,0]])

X = trainData[:,1:] # The input vector
X = X / X.sum(axis=1)[:, np.newaxis] # Normalize on a per row basis
n = 784 # Dimension of the input space (28 * 28 = 784 pixels)
m = 4 # Dimension of hidden layer ( 4 nodes )
l = 10 # Dimension of output ( 10 possible digits to classify )
k = 60000 # Size of training data

# Define functions
def ReLU(Z):
  return np.maximum(Z, 0)

def ReluPrime(Z):
  return Z > 0

def softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A

def error(l, p):
  if (len(p) != len(l)):
    raise ValueError("Lengths must be the same")
  return np.sum([0.5 * (p[i] - l[i])**2 for i in range(len(p))])


def initializeParameters():
  # Initialize weights for the input and the hidden layer
  # Between -0.5 and 0.5 since we are using ReLU as activation function
  W0 = np.random.rand(n, m) - 0.5
  W1 = np.random.rand(m, l) - 0.5
  b0 = np.random.rand(m) - 0.5
  b1 = np.random.rand(l) - 0.5

  return W0, b0, W1, b1

def forwardProp(W0, W1, b0, b1, X_i):
  A0 = W0.T.dot(X_i) + b0
  Z0 = ReLU(A0)
  Z1 = W1.T.dot(Z0) + b1
  P = softmax(Z1)

  if not math.isclose(np.sum(P), 1.0):
    raise ValueError("forwardProp: P does not sum to 1")

  return A0, P

def backProp(W1, P, X_i, Y_i, A0):

  def z(k):
    return (P[k] - Y_i[k]) * P[k] * (1 + P[k])
  
  dW0 = np.array([[ReluPrime(A0[j]) 
    * X_i[j] 
    * np.sum([z(k) * W1[j][k] for k in range(l)]) for i in range(n)] 
          for j in range(m)])

  db0 = np.array([
    ReluPrime(A0[i]) * 
    np.sum([z(k) * W1[i][k] for k in range(l)]) 
      for i in range(m)])

  dW1 =  np.array([[ReLU(A0[i]) 
    * z(j) 
      for i in range(m)] 
        for j in range(l)])

  db1 =  np.array([z(k) for k in range(l)])

  return dW0, db0, dW1, db1


def getPredictions(P):
    return np.argmax(P, 0)

def getAccuracy(p, Y):
  return np.sum(p == Y) / Y.size

def updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon):
  W0 = W0 - epsilon * dW0.T
  b0 = b0 - epsilon * db0
  W1 = W1 - epsilon * dW1.T
  b1 = b1 - epsilon * db1

  return W0, b0, W1, b1

def gradientDescent(X, Y, epsilon, iterations):
  
  W0, b0, W1, b1 = initializeParameters()
  errors = []
  for i in range(iterations):
    A0, P = forwardProp(W0, W1, b0, b1, X[i])
    dW0, db0, dW1, db1 = backProp(W1, P, X[i], Y[i], A0)
    W0, b0, W1, b1 = updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon)

    currError = error(P, Y[i])
    errors.append(currError)

    if i % 50 == 0:
      print("Iteration: ===================", i)
      print("Error: ", currError)

  plt.plot(errors)
  plt.ylabel("Errors")
  plt.savefig('errors.png')
  return W0, b0, W1, b1


W0, b0, W1, b1 = gradientDescent( X, Y, epsilon, 2)
# %%
