import argparse
import os
from codebase.utils import make_dir_if_not_exist
from codebase.load_config import load_dirs_config
from testing.track_decompositions import main as track_decompositions

parser = argparse.ArgumentParser(description='Plot decompositions through training')
parser.add_argument('-n', '--name', help='experiment name', default='')
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='madras-vector.json')
parser.add_argument('-fd', '--figdir', help='dir for figs', default='figs')
args = vars(parser.parse_args())

# get params
dirs = load_dirs_config(args) 
expdir = os.path.join(dirs['exp'], args['name'])
figdir = os.path.join(args['figdir'], args['name'])
make_dir_if_not_exist(figdir)

base_groups = {'treatment-shift': ['L', 'L_do', 'L_t_shift'],
               'value': ['V', 'V_sample', 'V_treat', 'V_star'],
               'regret': ['V_regret_databias', 'V_regret_fnlearn', 'V_regret_fnlearn_unbiased', 'V_regret_databias_optfn']}
a0_groups = {'A0-{}'.format(k): ['A0_{}'.format(m) for m in base_groups[k]] for k in base_groups}
a1_groups = {'A1-{}'.format(k): ['A1_{}'.format(m) for m in base_groups[k]] for k in base_groups}

track_groups = {**base_groups, **a0_groups, **a1_groups}
track_pairs = {k: {m: ('A0_{}'.format(m), 'A1_{}'.format(m)) for m in base_groups[k]} for k in base_groups}
settings = {'plot': {}}
track_decompositions(expdir, figdir, track_groups, track_pairs, settings)


