import argparse
import os
import numpy as np
from codebase.load_config import load_dirs_config, load_data_config
from dataproc.make_synth_dataset_basic import main as make_synth_dataset_basic
from dataproc.create_data_splits import main as create_data_splits
import dataproc.SynthDatasetCreator

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='madras-vector.json')
parser.add_argument('-data', '--dataset', help='dataset name', default='')
args = vars(parser.parse_args())

# get params
dirs = load_dirs_config(args) 
data_kwargs = load_data_config(args, dirs, generate=True)
datadir = dirs['data']
npzfilename = data_kwargs['npzfile']
fnames = {
        'data_dir_out': os.path.join(datadir, os.path.dirname(npzfilename)), # '/scratch/gobi1/madras/datasets/synth_treatments_basic',
        'data_fout_name': npzfilename #'synth_treatments_basic.npz'
        }

data_gen_args = data_kwargs['data_gen_args']
seed = data_gen_args.pop('seed')
data_creator_str = data_gen_args.pop('creator_class')
DataCreator = getattr(dataproc.SynthDatasetCreator, data_creator_str)
data_generator = DataCreator(seed, **data_gen_args) # num_data, xdim, mu0, mu1, sd0, sd1, p0) 
generated_data = data_generator.generate_dataset()
print([np.mean(generated_data[k]) for k in generated_data])
create_data_splits(fnames, seed, generated_data)
