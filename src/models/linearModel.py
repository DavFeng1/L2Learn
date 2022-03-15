import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.seterr(all='raise')

class LinearModel:
  """
    2 Variable linear regression model using gradient descent
  """
  def __init__(self):
    # Data 
    trainDataFrame = pd.read_csv("../datasets/linear-regression/linearTest.csv")
    trainData = np.array(trainDataFrame)
    self.X = trainData[:, 0]
    self.Y = trainData[:, 1]
    print("Loaded training data X {}, Y {}".format(self.X.shape, self.Y.shape))

    # Parameters
    self.b = random.random()
    self.m = random.random()
    print("Initialized linear model with parameters m={}, b={}".format(self.m, self.b))

    # Hyperparameters
    self.learning_rate = 0.0001
  
  def gradient_descent(self, epochs):
    error = []

    for i in range(epochs):
      prediction = self.m * self.X + self.b
      dm =  - 2 * np.dot(self.X, self.Y - prediction) / len(self.X)
      db =  - 2 * np.mean(self.Y - prediction)
      self.m = self.m - self.learning_rate * dm
      self.b = self.b - self.learning_rate * db

      curr_error = mean_square_error(prediction, self.Y)
      error.append(curr_error)
      if i % 50 == 0:
        print("Trianing epoch {}".format(i))

    plt.plot(error, color="red", label="Mean Squared Error")
    plt.title("Training error vs Epochs")
    plt.legend()
    plt.show()

  def predict(self):
    print("Making prediction with m={} b={}".format(self.m, self.b))
    Y_prediction = self.m * self.X + self.b
    error = mean_square_error(self.Y, Y_prediction)
    self.prediction = Y_prediction
    print("Made prediction with error: {}".format(error))
  
  def visualize(self):
    # Plot training error
    X = np.linspace(0, 10, 300)
    Y = self.m * X + self.b
    plt.title("Prediction") 
    plt.plot(self.X, self.prediction, color="black", label="prediction")
    plt.plot(self.X, self.Y, '.' , color="green", label="trianing data")
    plt.legend()
    plt.show()
  
def mean_square_error(y, y_hat):
  return np.sum((y - y_hat) ** 2).mean()


mdl = LinearModel()
mdl.gradient_descent(25)
mdl.predict()
mdl.visualize()

