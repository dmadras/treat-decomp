import argparse
import os
import json
from codebase.dataset import Dataset
from codebase.cf_models import *
from codebase.trainer import Trainer
from codebase.defaults import get_default_kwargs
from codebase.utils import make_dir_if_not_exist
from codebase.load_config import load_data_config, load_model_config, load_dirs_config

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
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='madras-vector.json')
args = vars(parser.parse_args())


# get params
dirs = load_dirs_config(args) 
data_kwargs = load_data_config(args, dirs)
model_kwargs = load_model_config(args, dirs)

#get dataset
data = Dataset(**data_kwargs)
print('Dataset loaded from {}.'.format(dirs['data']))

#get model
if args['model'] == 'BinaryCFMLP':
    model = BinaryCFMLP(**model_kwargs)
elif args['model'] == 'BinaryCFDoubleMLP':
    model = BinaryCFDoubleMLP(**model_kwargs)
else:
    raise Exception('bad model name')
print('Model loaded.')

with tf.Session() as sess:
    print('Session created.')
    resdirname = os.path.join(dirs['exp'], args['name'])
    logdirname = os.path.join(dirs['log'], args['name'], 'tb_log')
    ckptdirname = os.path.join(resdirname, 'checkpoints')
    for d in [resdirname, logdirname, ckptdirname]:
        make_dir_if_not_exist(d)

    #create Trainer
    trainer = Trainer(model, data, batch_size=args['batch_size'], sess=sess, logs_path=logdirname, \
                 checkpoint_path=ckptdirname, results_path=resdirname)
    save_path = trainer.train(n_epochs=args['num_epochs'], patience=args['patience'])
    trainer.restore(save_path)
    trainer.test()

#save args
args_path = os.path.join(resdirname, 'args.json')
json.dump(args, open(args_path, 'w'))
