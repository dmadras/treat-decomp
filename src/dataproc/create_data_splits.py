import os
import numpy as np
from dataproc.utils import save_tensors
from codebase.utils import make_dir_if_not_exist


def get_split_inds(n, test_pct, valid_pct):
    shuf = np.arange(n)
    np.random.shuffle(shuf)
    train_and_valid_pct = 1. - test_pct
    train_pct = train_and_valid_pct * (1. - valid_pct)
    valid_border = int(n * train_pct) + 1
    test_border = int(n * train_and_valid_pct) + 1
    train_inds = shuf[:valid_border]
    valid_inds = shuf[valid_border:test_border]
    test_inds = shuf[test_border:]
    assert sorted(np.concatenate([train_inds, valid_inds, test_inds])) == sorted(shuf)
    return train_inds, valid_inds, test_inds

def main(fnames, seed, save_dict):
    np.random.seed(seed)
    make_dir_if_not_exist(fnames['data_dir_out'])
    npzname_out = os.path.join(fnames['data_dir_out'], fnames['data_fout_name'])
    n = save_dict['X'].shape[0]
    train_inds, valid_inds, test_inds = get_split_inds(n, test_pct=0.3, valid_pct=0.2)
    split_save_dict = {}
    for inds, phase in [(train_inds, 'train'), (valid_inds, 'valid'), (test_inds, 'test')]:
        for t in save_dict:
            split_save_dict['{}_{}'.format(t, phase)] = save_dict[t][inds]
    save_tensors(split_save_dict, npzname_out)    


if __name__ == '__main__':
    import sys
    sys.path.append('src')
    fnames = {
            'DATA_DIR_IN':'/scratch/gobi1/madras/twins',
            'data_fname': 'twins_light_gestat_proxies.npz',
            'DATA_DIR_OUT':'/scratch/gobi1/madras/twins',
            'data_fout_name': 'twins_light_gestat_proxies_splits.npz',
            }
    seed = 0
    main(fnames, seed)
