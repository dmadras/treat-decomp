import argparse
import os
import subprocess
import json
import itertools

def get_num_tokens(curr_args_fname):
    f = open(curr_args_fname, 'r')
    n_tokens = len(f.readline().strip().split())
    f.close()
    return n_tokens

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-d', '--sweep_dir', help='sweep directory', default='')
parser.add_argument('-r', '--results_dir', help='dir where results will be stored', default='/scratch/gobi1/madras/treat-decomp')
parser.add_argument('-n', '--n_args', help='num simultaneous xargs', default=4, type=int)
parser.add_argument('-p', '--partition', help='request gpu or cpu?', default='gpu')

args = vars(parser.parse_args())

res_dir_name = args['results_dir']
sweep_file_name = os.path.join(args['sweep_dir'], 'sweep.json')
sweep_args = json.load(open(sweep_file_name, 'r'))
script_name = sweep_args.pop('script_name')
overrides = sweep_args.pop('overrides')
sweep_name = sweep_args['sweep_name']

# make name from only the keys with list values
s_keys = sorted(sweep_args.keys())
o_keys = sorted(overrides.keys())
list_keys = list(filter(lambda v: isinstance(sweep_args[v], list), s_keys))
single_keys = list(filter(lambda v: not isinstance(sweep_args[v], list), s_keys))
list_overrides = list(filter(lambda v: isinstance(overrides[v], list), o_keys))
single_overrides = list(filter(lambda v: not isinstance(overrides[v], list), o_keys))

opts = [sweep_args[k] for k in list_keys] 
opts_product = itertools.product(*opts)
ovs = [overrides[k] for k in list_overrides]
ovs_product = itertools.product(*ovs)
ttl_product = itertools.product(*[opts_product, ovs_product])

# store the arg values for each name
# store the override values for each name
# get base args
base_opts = [sweep_args[k] for k in single_keys] + [overrides[k] for k in single_overrides]
opts_dict = {}
for opt_list, ovs_list in ttl_product:
    name = '{}-'.format(sweep_name)
    name += ','.join(['{}={}'.format(list_keys[i], str(opt_list[i])) for i in range(len(opt_list))]) + ','
    name += ','.join(['{}={}'.format(list_overrides[i], str(ovs_list[i])) for i in range(len(ovs_list))])
    opts_dict[name] ={'opts': dict(zip(list_keys, opt_list)), 'ovs': dict(zip(list_overrides, ovs_list)), 
                        'base': dict(zip(single_keys + single_overrides, base_opts))}

#open file to store commands
args_file_name = os.path.join(args['sweep_dir'], 'args.txt')
dirs_file_name = os.path.join(args['sweep_dir'], 'dirs.txt')
args_f = open(args_file_name, 'w')
dirs_f = open(dirs_file_name, 'w')
for cmd_name in opts_dict:
    s = 'python {} -n {} '.format(script_name, cmd_name)
    opts = opts_dict[cmd_name]['opts']
    for o_name in list_keys:
        s += '--{} {} '.format(o_name, opts[o_name])
    base_opts = opts_dict[cmd_name]['base']
    for o_name in single_keys:
        s += '--{} {} '.format(o_name, base_opts[o_name])
    ovs = opts_dict[cmd_name]['ovs']
    s += '--overrides '
    for o_name in list_overrides:
        s += '{}={},'.format(o_name, ovs[o_name])
    for o_name in single_overrides:
        s += '{}={},'.format(o_name, base_opts[o_name])
    s = s[:-1]
    # print commands to file
    args_f.write(s + '\n')
    #write expname to file
    dirs_f.write(cmd_name + '\n')
args_f.close()
dirs_f.close()

# make go.sh file

if args['partition'] == 'gpu':
    srun_opts = '--gres=gpu:1 -p gpu'
elif args['partition'] == 'cpu':
    srun_opts = '-c 1'

go_file_name = os.path.join(args['sweep_dir'], 'go.sh')
go_f = open(go_file_name, 'w')
n_tokens = get_num_tokens(args_file_name)
go_cmd = "xargs -n {:d} -P {:d} srun {} < {}\n".format(n_tokens, args['n_args'], srun_opts, args_file_name)
go_f.write(go_cmd)
go_f.close()
os.system('chmod 744 {}'.format(go_file_name))
print('Saved script to {}'.format(go_file_name))
