import os

def get_default_kwargs(opts, dirs):
    """some defaults kwargs dicts for declaring models and datasets, e.g.,

    >>> data_kwargs, model_kwargs = get_default_kwargs('twins')
    >>> data = MyDataset(**data_kwargs)
    >>> model = MyModel(**model_kwargs)

    """
    dataset_agnostic_data_kwargs = dict(
        seed=opts['data_random_seed']
    )
    data_kwargs = dict(
	    twins=dict(
                name='twins',
                npzfile=os.path.join(dirs['data'], 'twins/twins_light_proxies_splits.npz')
            ),
	    synth=dict(
		name='synth',
		npzfile=(os.path.join(dirs['data'], 'synth_cf/synth_cf_splits.npz'))
	    )
    )
    dataset_agnostic_model_kwargs = dict(
        seed=opts['model_random_seed'],
        ydim=1,
        tdim=1
    )
    model_kwargs = dict(
	    twins=dict(
		xdim=454,
		hidden_layer_specs={
		    'layer_sizes': [15 for l in range(opts['num_hid_layers'])],
		    'activ': 'relu'
		}
	    ),
	    synth=dict(
		xdim=6,
		hidden_layer_specs={
                    'layer_sizes': [3 for l in range(opts['num_hid_layers'])],
		    'activ': 'relu'
		}
	    )
    )
    return ({**data_kwargs[opts['dataset']], **dataset_agnostic_data_kwargs},\
	    {**model_kwargs[opts['dataset']], **dataset_agnostic_model_kwargs})

