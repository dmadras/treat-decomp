import json
import os

CONFIG_DIR = 'confs'

def load_dirs_config(args):
    dirs_config_path = os.path.join(CONFIG_DIR, 'dirs', args['dirconf'])
    dirs = json.load(open(dirs_config_path, 'r'))
    return dirs

def load_data_config(args, dirs, generate, overrides=''):
    data_config_name = '{}.json'.format(args['dataset'])
    data_config_path = os.path.join(CONFIG_DIR, 'data', data_config_name)
    data_config_opts = json.load(open(data_config_path, 'r'))
    data_config_opts = override(data_config_opts, overrides, 'data')
    npz_path = os.path.join(dirs['data'], data_config_opts['npzfile'])
    if not generate:
        data_kwargs = {
            'name': data_config_opts['name'],
            'npzfile': npz_path.format(*[args[a] for a in data_config_opts["npz_format_args"]]),
            'seed': args['data_random_seed']
            }
    else:
        data_gen_args = data_config_opts['data_generation_args']
        data_gen_args = override(data_gen_args, overrides, 'datagen')
        data_kwargs = {
            'name': data_config_opts['name'],
            'npzfile': npz_path.format(*[args[a] for a in data_config_opts["npz_format_args"]]),
            'data_gen_args': data_gen_args 
            }
    return data_kwargs

def load_model_config(args, dirs, overrides=''):
    model_config_name = '{}.json'.format(args['dataset'])
    model_config_path = os.path.join(CONFIG_DIR, 'model', model_config_name)
    model_config_opts = json.load(open(model_config_path, 'r'))
    model_config_opts = override(model_config_opts, overrides, 'model')
    model_kwargs = {
        'xdim': model_config_opts['xdim'],
        'ydim': 1,
        'tdim': 1,
        'adim': 1,
        'hidden_layer_specs': {
            'activ': model_config_opts['activ'],
            'layer_sizes': [model_config_opts['num_hid_units'] for layer in range(args['num_hid_layers'])] 
            },
        'seed': args['model_random_seed']
        }
    return model_kwargs

def override(main_kwargs, overrides, override_key):
    ov_list = [o.split('=') for o in overrides.split(',')] if overrides != '' else []
    print(ov_list)
    ov_info = {o[0]: o[1] for o in ov_list}
    print(ov_info, ov_list)
    for k in ov_info:
        k_info = k.split('.')
        if k_info[0] == override_key:
            assert k_info[1] in main_kwargs
            main_kwargs[k_info[1]] = convert_to_type(ov_info[k], main_kwargs[k_info[1]])
    return main_kwargs

def convert_to_type(source, target):
    if isinstance(target, int): return int(source)
    elif isinstance(target, float): return float(source)
    elif isinstance(target, str): return str(source)
    else: 
        raise Exception('What is {}? Cant convert {} to this type'.format(target, source))

