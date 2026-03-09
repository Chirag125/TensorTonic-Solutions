import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    l = np.asarray(x,dtype = float)
    s = 1/(1+ np.exp(-l))
    return s
  