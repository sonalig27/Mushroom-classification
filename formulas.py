import math
import numpy as np
def sig(x):
    #use logistic function as activation function
    return 1.0 / 1.0 + np.exp(-x)

def inv_sig(x):
    #derivative of the output of neuron with respect to its input
    return sig(x)*(1-sig(x))
    #return np.exp(-x)/((1 + np.exp(-x))**2)

def err(o, t):
    #squared error function, o is the actual output value and t is the target output
    return np.mean((o - t) ** 2)
    # # Retrieving number of samples in dataset
    # number_samples = len(o)
    # # Summing square differences between predicted and expected values
    # accumulated_error = 0.0
    # for i in range (number_samples):
    #     accumulated_error += (t - o) ** 2
    # # Calculating mean and dividing by 2
    # mse_error = (1.0 / (2 * number_samples)) * accumulated_error
    # return mse_error
def inv_err(o, t):
    #derivative of squared error function with respect to o
    return np.mean(2*(o-t))


