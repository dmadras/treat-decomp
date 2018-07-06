from dataproc.make_synth_dataset_basic import main as make_synth_dataset_basic
from dataproc.create_data_splits import main as create_data_splits

seed = 0
fnames = {
        'data_dir_out': '/scratch/gobi1/madras/synth_treatments_basic',
        'data_fout_name': 'synth_treatments_basic.npz'
        }

num_data = 10000
xdim = 10
mu0 = -1
mu1 = 1
sd0 = 2
sd1 = 2
p0 = 0.7
generated_data = make_synth_dataset_basic(seed, num_data, xdim, mu0, mu1, sd0, sd1, p0)
create_data_splits(fnames, seed, generated_data)
