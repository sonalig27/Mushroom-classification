import numpy as np
import random
from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err
curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        # evaluation part
        # Get input, compute the output of layer nodes.
        self.A = {}
        self.H = {}
        self.H[0] = self.input_vals.reshape(1, -1)
        self.A[self.layer_num] = np.matmul(self.H[self.layer_num],
                                           self.weight[self.layer_num]) + self.bias[self.layer_num]
        self.H[self.layer_num] = sig(self.A[self.layer_num])
        return self.H[self.layer_num]

    def backprop(self, other):
        # use backpropagation method to update weights
        self.eval()
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        self.dA[self.layer_num] = (self.H[self.layer_num] - other)
        for k in range(2, 0, -1):
            self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k - 1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k - 1] = np.multiply(self.dH[k - 1], inv_sig(self.H[k - 1]))


class cfile(file):
    def __init__(self, name, mode='r'):
        self = file.__init__(self, name, mode)

    def w(self, string):
        self.writelines(str(string) + '\n')
        return None
