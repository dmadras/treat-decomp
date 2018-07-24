import argparse
import os
from codebase.main import main
from dataproc.make_synth_dataset import main as datagen_main
from codebase.load_config import load_data_config, load_model_config, load_dirs_config, override

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-d', '--dataset', help='dataset name', default='')
parser.add_argument('-n', '--name', help='experiment name', default='temp')
parser.add_argument('-m', '--model', help='model name', default='BinaryCFMLP')
parser.add_argument('-ne', '--num_epochs', help='number of training epochs', default=10000, type=int)
parser.add_argument('-nh', '--num_hid_layers', help='number of hidden layers', default=1, type=int)
parser.add_argument('-pa', '--patience', help='training patience for early stopping', default=10, type=int)
parser.add_argument('-bs', '--batch_size', help='batch size for training', default=64, type=int)
parser.add_argument('-dseed', '--data_random_seed', help='data random seed', default=0, type=int)
parser.add_argument('-mseed', '--model_random_seed', help='model random seed', default=0, type=int)
parser.add_argument('-rseed', '--random_seed', help='data and model random seed', default=None, type=int)
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='madras-vector.json')
parser.add_argument('-o', '--overrides', help='what variables to override', default='')
parser.add_argument('-swn', '--sweep_name', help='the name of this sweep', default=None)
args = vars(parser.parse_args())


# get params
dirs = load_dirs_config(args, sweep_name=args['sweep_name']) 
data_kwargs_for_gen = load_data_config(args, dirs, generate=True, overrides=args['overrides'])
data_gen_args = data_kwargs_for_gen['data_gen_args']
datadir = dirs['data']
npzfilename = data_kwargs_for_gen['npzfile']
fnames = {
        'data_dir_out': os.path.join(datadir, os.path.dirname(npzfilename)), # '/scratch/gobi1/madras/datasets/synth_treatments_basic',
        'data_fout_name': npzfilename #'synth_treatments_basic.npz'
        }

#adjust seeds
if not args['random_seed'] is None:
    args['data_random_seed'] = args['random_seed']
    args['model_random_seed'] = args['random_seed']

#run data generation
datagen_main(data_gen_args, fnames)

#get params for model
data_kwargs = load_data_config(args, dirs, generate=False, overrides=args['overrides'])
model_kwargs = load_model_config(args, dirs, overrides=args['overrides'])

#run model
main(args, dirs, data_kwargs, model_kwargs)

