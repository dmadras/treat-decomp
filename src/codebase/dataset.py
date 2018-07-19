import numpy as np
import collections
from codebase.utils import switch
EPS = 1e-12

class Dataset(object):

    def __init__(self, name, npzfile, seed=0, load_on_init=True):
        self.name = name
        self.npzfile = npzfile
        self.loaded = False
        self.seed = seed
        if load_on_init:
            self.load()
            # self.centre_and_scale()

    def load(self):
        if not self.loaded:
            dat = np.load(self.npzfile)
            self.dat = dat
            phases = ['train', 'valid', 'test']
            self.name_mapping = {'x': 'X', 'y_values': 'Y', 'a': 'A', 't_f': 'T_f', 't_cf': 'T_cf', 'y0': 'y0','y1': 'y1',  \
                    'y_f': 'Y_f', 'y_cf': 'Y_cf', 'z': 'Z', 'bayes_f': 'bayes_f', 'bayes_cf': 'bayes_cf', 't_prob': 'T_prob', \
                            'x_unb': 'X_unb', 'y_values_unb': 'Y_unb', 'a_unb': 'A_unb', 't_f_unb': 'T_f_unb', 't_cf_unb': 'T_cf_unb', \
                            'y_f_unb': 'Y_f_unb', 'y_cf_unb': 'Y_cf_unb', 'z_unb': 'Z_unb', 'bayes_f_unb': 'bayes_f_unb', 'bayes_cf_unb': 'bayes_cf_unb', \
                            't_prob_unb': 'T_prob_unb'}
            self.tensors = {name: {phase: dat['{}_{}'.format(self.name_mapping[name], phase)] for phase in phases} for name in self.name_mapping}
            self.loaded = True

    def get_batch_iterator(self, phase, mb_size):
        names = list(self.name_mapping.keys())
        tensors = [self.tensors[name][phase] for name in names]
        sz = tensors[0].shape[0]
        batch_inds = make_batch_inds(sz, mb_size, self.seed, phase)
        iterator = DatasetIterator(tensors, names,  batch_inds)
        return iterator

    # def centre_and_scale(self):
    #     x_means = np.mean(self.x['train'], axis=0)
    #     x_sds = np.std(self.x['train'], axis=0)
    #     for phase in ['train', 'valid', 'test']:
    #         self.x[phase] = centre_and_scale(self.x[phase], x_means, x_sds)

class DatasetIterator(collections.Iterator):

    def __init__(self, tensor_list, name_list, ind_list):
        self.tensors = tensor_list
        self.names = name_list
        self.inds = ind_list
        self.curr = 0
        self.ttl_minibatches = len(self.inds)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.ttl_minibatches:
            raise StopIteration
        else:
            inds = self.inds[self.curr]
            minibatch = {self.names[t]: self.tensors[t][inds] for t in range(len(self.tensors))}
            self.curr += 1
            return minibatch

def make_batch_inds(n, mb_size, seed=0, phase='train'):
    np.random.seed(seed)
    if phase == 'train':
        shuf = np.random.permutation(n)
    else:
        shuf = np.arange(n)
    start = 0
    mbs = []
    while start < n:
        end = min(start + mb_size, n)
        mb_i = shuf[start:end]
        mbs.append(mb_i)
        start = end
    return mbs

def centre_and_scale(x, mn, sd):
    centred_x = x - mn
    scaled_x = centred_x / np.maximum(sd, EPS)
    return scaled_x
