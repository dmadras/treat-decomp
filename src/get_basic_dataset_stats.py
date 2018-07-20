import numpy as np
import argparse
import os
from codebase.dataset import Dataset
from codebase.utils import switch
from codebase.metrics import subgroup
from codebase.load_config import load_dirs_config, load_data_config

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='madras-vector.json')
parser.add_argument('-data', '--dataset', help='dataset name', default='')
args = vars(parser.parse_args())

# get params
dirs = load_dirs_config(args) 
data_kwargs = load_data_config(args, dirs, generate=True)
datadir = dirs['data']
npzfilename = data_kwargs['npzfile']

dat = Dataset(args['dataset'], npzfilename) 
tensor_names = ['y_f', 'y_cf', 't_f', 'y0', 'y1', 'z']
for tnm in tensor_names:
    res = np.mean(dat.tensors[tnm]['train'])
    print('Avg {}: {:.3f}'.format(tnm, res))
    res = subgroup(np.mean, dat.tensors['a']['train'], [dat.tensors[tnm]['train']]) 
    print('Avg {} (A = 1): {:.3f}'.format(tnm, res))
    res = subgroup(np.mean, 1 - dat.tensors['a']['train'], [dat.tensors[tnm]['train']]) 
    print('Avg {} (A = 0): {:.3f}'.format(tnm, res))
