import json
import os

CONFIG_DIR = 'confs'

def load_dirs_config(args):
    dirs_config_path = os.path.join(CONFIG_DIR, 'dirs', args['dirconf'])
    dirs = json.load(open(dirs_config_path, 'r'))
    return dirs

def load_data_config(args, dirs):
    data_config_name = '{}.json'.format(args['dataset'])
    data_config_path = os.path.join(CONFIG_DIR, 'data', data_config_name)
    data_config_opts = json.load(open(data_config_path, 'r'))
    npz_path = os.path.join(dirs['data'], data_config_opts['npzfile'])
    data_kwargs = {
        'name': data_config_opts['name'],
        'npzfile': npz_path.format(*[args[a] for a in data_config_opts["npz_format_args"]]),
        'seed': args['data_random_seed']
        }
    return data_kwargs

def load_model_config(args, dirs):
    model_config_name = '{}.json'.format(args['dataset'])
    model_config_path = os.path.join(CONFIG_DIR, 'model', model_config_name)
    model_config_opts = json.load(open(model_config_path, 'r'))
    model_kwargs = {
        'xdim': model_config_opts['xdim'],
        'ydim': 1,
        'tdim': 1,
        'hidden_layer_specs': {
            'activ': model_config_opts['activ'],
            'layer_sizes': [model_config_opts['num_hid_units'] for layer in range(args['num_hid_layers'])] 
            },
        'seed': args['model_random_seed']
        }
    return model_kwargs


