import numpy as np
from dataproc.make_synth_dataset_basic import main as make_synth_dataset_basic
from dataproc.create_data_splits import main as create_data_splits
import dataproc.SynthDatasetCreator

def main(data_gen_args, fnames):
    seed = data_gen_args.pop('seed')
    data_creator_str = data_gen_args.pop('creator_class')
    DataCreator = getattr(dataproc.SynthDatasetCreator, data_creator_str)
    data_generator = DataCreator(seed, **data_gen_args) # num_data, xdim, mu0, mu1, sd0, sd1, p0) 
    generated_data = data_generator.generate_dataset()
    print([np.mean(generated_data[k]) for k in generated_data])
    create_data_splits(fnames, seed, generated_data)
