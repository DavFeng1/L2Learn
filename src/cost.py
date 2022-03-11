import numpy as np

def meanSquareError(P, Y):
  if (P.shape != Y.shape):
    raise ValueError("Lengths must be the same")

  return 0.5 * np.sum((P - Y) ** 2)