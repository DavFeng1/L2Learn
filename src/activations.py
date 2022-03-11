import numpy as np

def sigmoid(x):
  return 1 / ( 1 + np.exp(-x))

def sigmoidPrime(x):
  return np.exp(-x) / ( 1 + np.exp(-x)) ** 2

def ReLU(Z):
  return np.maximum(Z, 0)

def ReluPrime(Z):
  return 1 * ( Z > 0 )

def softmax(Z):
  numerator = np.exp(Z)
  denominator = sum(np.exp(Z))
  return numerator / denominator
