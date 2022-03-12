import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functions import sigmoid, sigmoidPrime, softmax, meanSquareError, getAccuracy
np.seterr(all='raise')

# =============================================================================
num_inputs = 784 # Dimension of the input space (28 * 28 = 784 pixels)
num_hidden = 4 # Dimension of hidden layer ( 4 nodes )
num_labels = 10 # Dimension of output ( 10 possible digits to classify )
training_batch_size = 1000 # Size of training data
learning_rate = 0.1
precision = 1e-7

def initializeTestData():
  testDataFrame = pd.read_csv("./datasets/mnist/mnistTest.csv")
  testData = np.array(testDataFrame)

  Y_test = oneHotEncode(testData[:, 0])  # (60000, 10)
  X_test = testData[:, 1:] / 255 # (60000, 784)

  print("Initialized testing data: ")
  print("X shape: {}".format(X_test.shape))
  print("Y shape: {}".format(Y_test.shape))

  return X_test, Y_test

def initializeTrainData():
  trainDataFrame = pd.read_csv("./datasets/mnist/mnistTrain.csv")
  trainData = np.array(trainDataFrame)

  Y_train = oneHotEncode(trainData[:training_batch_size, 0])  # (training_batch_size, 10)
  X_train = trainData[:training_batch_size, 1:] / 255 # (training_batch_size, 784)

  print("Initialized training data: ")
  print("X shape: {}".format(X_train.shape))
  print("Y shape: {}".format(Y_train.shape))

  return X_train, Y_train

def oneHotEncode(Y):
  return np.array([[1 if label == i else 0 for i in range(10)] for label in Y])

def initializeParameters():
  W0 = np.random.rand(num_hidden, num_inputs) - 0.5
  b0 = np.random.rand(num_hidden) - 0.5
  W1 = np.random.rand(num_labels, num_hidden) - 0.5
  b1 = np.random.rand(num_labels) - 0.5

  return W0, b0, W1, b1

  
def forwardProp(W0, W1, b0, b1, X):
  Z0 = np.dot(W0, X)
  A0 = sigmoid(Z0)
  Z1 = W1.dot(A0) + b1
  P = softmax(Z1, precision).T

  return Z0.T, P

def backProp(W1, P, X, Y, Z0):
  PY = (P - Y) * P * (1 + P)

  dW0 = np.outer(sigmoidPrime(Z0).dot(W1.T.dot(PY)), X)
  db0 = sigmoidPrime(Z0) * W1.T.dot(PY)
  dW1 = np.outer(PY, sigmoid(Z0))
  db1 = PY

  return dW0, db0, dW1, db1


def updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon):
  W0 = W0 - epsilon * dW0
  b0 = b0 - epsilon * db0
  W1 = W1 - epsilon * dW1
  b1 = b1 - epsilon * db1

  return W0, b0, W1, b1

def gradientDescent(epsilon, num_epochs):

  X, Y = initializeTrainData()
  W0, b0, W1, b1 = initializeParameters()

  training_accuracy = []
  training_error = []

  print("Training... Batch Size: {}, Epochs: {}".format(training_batch_size, num_epochs))

  for i in range(num_epochs):

    if i % 100 == 0:
      print("Epoch: {}/{}".format(i, num_epochs))

    P_total = np.zeros((training_batch_size, num_labels))

    for j in range(training_batch_size):

      Z0, P = forwardProp(W0, W1, b0, b1, X[j])
      dW0, db0, dW1, db1 = backProp(W1, P, X[j], Y[j], Z0)
      W0, b0, W1, b1 = updateParams(W0, W1, b0, b1, dW0, dW1, db0, db1, epsilon)

      P_total[j] = P

    curr_error = meanSquareError(P_total, Y, training_batch_size)
    training_error.append(curr_error)
      
    epoch_acc = getAccuracy(P_total, Y)
    training_accuracy.append(epoch_acc)

  # Plot training error
  plt.plot(training_error, label="error")
  plt.title("Training Error")
  plt.savefig("training_error.png")
  plt.close()

  # Plot training accuracy
  plt.plot(training_accuracy, label="Accuracy")
  plt.title("Training Accuracy")
  plt.savefig("training_accuracy.png")
  plt.close()

  return W0, b0, W1, b1

def testModel(W0, b0, W1, b1):
  X_test, Y_test = initializeTestData();
  n = len(X_test)

  P_total = np.zeros((n, num_labels))

  for i in range(len(X_test)):
    Z0, P = forwardProp(W0, W1, b0, b1, X_test[i])
    P_total[i] = P

  accuracy = getAccuracy(P_total, Y_test)

  print("Model achieved {}% accuracy over {} test examples".format(accuracy, n))
  return accuracy

def trainAndTest(learning_rate, epochs):
  print("Train and test...")
  W0, b0, W1, b1 = gradientDescent(learning_rate, epochs)
  testModel(W0, b0, W1, b1)

trainAndTest(learning_rate, epochs=300)