import os
import numpy as np

def make_dir_if_not_exist(d):
    if not os.path.exists(d):
        os.makedirs(d)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def switch(x0, x1, s):
    return np.multiply(x0, 1. - s) + np.multiply(x1, s)
