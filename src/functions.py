import numpy as np

def sigmoid(x):
  return 1 / ( 1 + np.exp(-x))

def sigmoidPrime(x):
  return np.exp(-x) / ( 1 + np.exp(-x)) ** 2

def ReLU(Z):
  return np.maximum(Z, 0)

def ReluPrime(Z):
  return 1 * ( Z > 0 )

def softmax(Z, eps):
  Z[np.abs(Z) < eps] = 0
  ans = np.exp(Z) / sum(np.exp(Z))
  ans[np.abs(ans) < eps] = 0
  return ans

def getAccuracy(P, Y):
  """
    Get Per-Epoch accuracy
  """
  if P.shape != Y.shape:
    raise ValueError("P and Y sizes do not match")
  predictions = np.argmax(P)
  expectations = np.argmax(Y)
  return np.sum(predictions == expectations) / expectations.size


def meanSquareError(P, Y, n):
  if (P.shape != Y.shape):
    raise ValueError("Lengths must be the same")

  ans = 0
  for i in range(n):
    ans += 0.5 * np.sum((P[i] - Y[i]) ** 2)

  return ans / n