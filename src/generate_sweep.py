import argparse
import os
import subprocess

def get_args_fname(nm, i):
    return '{}_{:d}.txt'.format(nm, i)

def get_num_tokens(curr_args_fname):
    f = open(curr_args_fname, 'r')
    n_tokens = len(f.readline().strip().split())
    f.close()
    return n_tokens

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-d', '--sweep_dir', help='sweep directory', default='')
parser.add_argument('-r', '--results_dir', help='dir where results will be stored', default='/scratch/gobi1/madras/causal-fairness')
parser.add_argument('-n', '--n_args', help='num simultaneous xargs', default=4, type=int)
parser.add_argument('-l', '--n_loops', help='number of loops in sweep', default=1, type=int)
parser.add_argument('-p', '--partition', help='request gpu or cpu?', default='gpu')

args = vars(parser.parse_args())

res_dir_name = args['results_dir']
sweep_file_name = os.path.join(args['sweep_dir'], 'sweep.sh')
args_file_name = os.path.join(args['sweep_dir'], 'args')
dirs_file_name = os.path.join(args['sweep_dir'], 'dirs')
for i in range(args['n_loops']):
    os.system('rm -f {}'.format(get_args_fname(args_file_name, i)))
    os.system('rm -f {}'.format(get_args_fname(dirs_file_name, i)))
cmd = './{} {} {} {}'.format(sweep_file_name, args_file_name, dirs_file_name, res_dir_name)
print('Running {}'.format(cmd))
os.system(cmd)

if args['partition'] == 'gpu':
    srun_opts = '--gres=gpu:1 -p gpu'
elif args['partition'] == 'cpu':
    srun_opts = '-c 1'

go_file_name = os.path.join(args['sweep_dir'], 'go.sh')
go_f = open(go_file_name, 'w')
for i in range(args['n_loops']):
    curr_args_fname = get_args_fname(args_file_name, i)
    n_tokens = get_num_tokens(curr_args_fname)
    go_cmd = "xargs -n {:d} -P {:d} srun {} < {}\n".format(n_tokens, args['n_args'], srun_opts, curr_args_fname)
    go_f.write(go_cmd)
go_f.close()
os.system('chmod 744 {}'.format(go_file_name))
print('Saved script to {}'.format(go_file_name))
