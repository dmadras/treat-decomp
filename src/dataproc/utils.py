import numpy as np


def save_tensors(save_dict, fnm):
    np.savez(fnm, **save_dict)
    # print('Tensor shapes:')
    # for t in save_dict: print(t, save_dict[t].shape)
    print('Saved tensors to {}'.format(fnm))

def write_headers(headers, fnm):
    f = open(fnm, 'w')
    for i in range(len(headers)):
        f.write('{:d}: {}\n'.format(i, headers[i]))
    f.close()
    print('Saved headers to {}'.format(fnm))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def generate_normal_data(mu, sd, n, xdim):
    return np.random.normal(mu, sd, size=(n, xdim))

def generate_bernoulli_data(p, n, dim):
    return np.random.binomial(1, p, size=(n, dim))

