import numpy as np
import pandas as pd
from matplotlib import pyplot as plot

def loadData():

  testData = pd.read_csv("./datasets/mnist/mnistTest.csv")

  print("Loaded test data")

  trainData = pd.read_csv("./datasets/mnist/mnistTrain.csv")

  print("Loaded training data")

  test = np.array(testData)

  print("Test data shape")
  print(test.shape)

  test2 = np.array([[[1], [2], [3]]])






