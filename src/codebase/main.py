import os
import json
from codebase.dataset import Dataset
from codebase.cf_models import *
from codebase.trainer import Trainer
from codebase.defaults import get_default_kwargs
from codebase.utils import make_dir_if_not_exist

def main(args, dirs, data_kwargs, model_kwargs):
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
