import numpy as np
import tensorflow as tf
import os
from codebase.metrics import *
from codebase.utils import make_dir_if_not_exist
from codebase.decompositions import calculate_policy_decomposition, calculate_outcome_prediction_decomposition

# defaults
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LOG_PATH = './tfboard_logs'
CHECKPOINT_PATH = None
RESULTS_PATH = None
MAX_EPOCH = 1000000

class Trainer(object):
    def __init__(self, model, data, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, sess=None, logs_path=LOG_PATH, \
                 checkpoint_path=CHECKPOINT_PATH, results_path=RESULTS_PATH):
        self.data = data
        if not self.data.loaded:
            self.data.load()
        self.model = model
        self.batch_size = batch_size
        self.batches_seen = 0
        self.logs_path = logs_path
        self.checkpoint_path = checkpoint_path
        self.results_path = results_path

        self.train_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)  
        self.train_op = self.train_opt.minimize(
            self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        )

        self.sess = sess or tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # having the Trainer manage saving enables checkpointing model params during training

    def process_minibatch(self, phase, feed_dict, losses, tensors, print_grad=False):
        loss_names = sorted(losses.keys())
        tensor_names = sorted(tensors.keys())
        ops = [losses[name] for name in loss_names] + [tensors[name] for name in tensor_names]
        if phase == 'train':
            ops = [self.train_op] + ops
            ret = self.sess.run(ops, feed_dict=feed_dict)
            ret = ret[1:]
        else:
            ret = self.sess.run(ops, feed_dict=feed_dict)
        loss_dict = {loss_names[i]: np.mean(ret[i]) for i in range(len(loss_names))}
        tensor_dict = {tensor_names[i]: ret[i + len(loss_names)] for i in range(len(tensor_names))}
        return loss_dict, tensor_dict

    def make_feed_dict(self, minibatch, bias_type='normal'):
        feed_pairs = {'x': self.model.X, 'y_f': self.model.Y,
                'y_cf': self.model.Y_cf, 't_f': self.model.T, 'a': self.model.A}
        if bias_type == 'normal':
            return {feed_pairs[k]: minibatch[k] for k in feed_pairs}
        if bias_type == 'flipped':
            return {feed_pairs[k]: minibatch[k + '_unb'] for k in feed_pairs}
        if bias_type == 'unbiased':
            return {feed_pairs[k]: get_unbiased_tensor(minibatch, k) for k in feed_pairs}

    def get_tracking_tensor_dict(self, minibatch, track_names, bias_type='normal'):
        if bias_type == 'normal':
            return {name: minibatch[name] for name in track_names}
        if bias_type == 'flipped':
            return {name: minibatch[name + '_unb'] for name in track_names}
        elif bias_type == 'unbiased':
            return {name: get_unbiased_tensor(minibatch, name) for name in track_names}

    def process_epoch(self, phase, tracker, epoch, bias_type='normal'):
        losses, tensors = tracker.losses, tracker.tensors
        epoch_iter = self.data.get_batch_iterator(phase, self.batch_size)
        L = {l: 0. for l in losses}
        T = {t: None for t in tensors}
        T_track = {t: None for t in tracker.track_tensor_names}
        self.batches_seen = 0
        for minibatch in epoch_iter:
            self.batches_seen += 1
            feed_dict = self.make_feed_dict(minibatch, bias_type=bias_type)
            loss_dict, tensor_dict = self.process_minibatch(phase, feed_dict, losses, tensors)
            L = {k: L[k] + loss_dict[k] for k in L}
            T = {k: np.concatenate((T[k], tensor_dict[k])) if not T[k] is None else tensor_dict[k] for k in T}
            T_track_minibatch = self.get_tracking_tensor_dict(minibatch, tracker.track_tensor_names, bias_type=bias_type)
            T_track = {k: np.concatenate((T_track[k], T_track_minibatch[k])) if not T_track[k] is None \
                            else T_track_minibatch[k] for k in T_track}
        for k in L: L[k] /= self.batches_seen
        T = {**T, **T_track}
        return L, T

    def get_metrics(self, tensors, metrics):
        met_dict = {}
        for m in metrics:
            m_fn = metrics[m]
            met_dict[m] = m_fn(tensors)
        return met_dict

    def filter_keys(self, d):
        return list(filter(lambda k: not 'A=0' in k and not 'A=1' in k, d.keys()))

    def create_res_str(self, epoch, L, M):
        ep_str = 'E{:d}: '.format(epoch) if not epoch is None else 'Final: '
        res_str = ep_str + ', '.join(['{}:{:.3f}'.format(l, L[l]) for l in self.filter_keys(L)]) 
                        # + ', ' + ', '.join(['{}:{:.3f}'.format(m, M[m]) for m in self.filter_keys(M)])
        return res_str

    def run_epoch_and_get_metrics(self, phase, tracker, epoch, bias_type='normal'):
        L, T = self.process_epoch(phase, tracker, epoch, bias_type=bias_type)
        M = self.get_metrics(T, tracker.metrics)
        res_str = self.create_res_str(epoch if phase != 'test' else None, L, M)
        return L, T, M, res_str

    def train(self, n_epochs, patience):
        min_val_loss, min_epoch = np.finfo(np.float32).max, -100
        tracker = self.model.tracker
        if not self.logs_path is None:
            # Create new Summary objects - move this to external file maybe
            summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)
        train_metrics_log = os.path.join(self.results_path, 'train_results.csv')
        valid_metrics_log = os.path.join(self.results_path, 'valid_results.csv')

        save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
        for epoch in range(n_epochs):
            train_L, train_T, train_metrics, train_res_str = self.run_epoch_and_get_metrics('train', tracker, epoch)
            valid_L, valid_T, valid_metrics, valid_res_str = self.run_epoch_and_get_metrics('valid', tracker, epoch)

            #do decomposition tracking on test set
            test_decomp_results = self.get_loss_decompositions('test', tracker, epoch)

            #do tensorboard tracking (maybe)
            if not self.logs_path is None:
                summary = tf.Summary()
                for l in tracker.losses.keys():
                    summary.value.add(tag=l, simple_value=valid_L[l])
                for m in tracker.base_metrics:
                    summary.value.add(tag=m, simple_value=valid_metrics[m])
                for d in test_decomp_results:
                    summary.value.add(tag=d, simple_value=test_decomp_results[d])
                    
                summary_writer.add_summary(summary, epoch)
                summary_writer.flush()

            #do a simple command line print
            msg = 'Train: {} | Valid: {}'.format(train_res_str, valid_res_str)
            print(msg)
            self.save_results(train_L, {**train_metrics, **test_decomp_results}, train_metrics_log, write_headers=epoch == 0)
            self.save_results(valid_L, {**valid_metrics, **test_decomp_results}, valid_metrics_log, write_headers=epoch == 0)
            # print('Metrics saved to {}'.format(train_metrics_log))
            # print('Metrics saved to {}'.format(valid_metrics_log))

            #if min validation loss, checkpoint model
            l = valid_L['loss']
            if l < min_val_loss:
                min_val_loss = l
                min_epoch = epoch
                if not self.checkpoint_path is None:
                    save_path = self.saver.save(self.sess, save_path) #Checkpoint
                    print('Best validation loss so far of {:.3f}, model saved to {}'.format(l, save_path))

            #if past patience, return for testing
            if epoch == n_epochs - 1 or epoch - min_epoch >= patience:
                print('Finished training: min validation loss was {:.3f} in epoch {:d}'.format(min_val_loss, min_epoch))
                break

        return save_path

    def get_loss_decompositions(self, phase, tracker, epoch):
        pred_results = {}
        for bias_type in ['normal', 'flipped', 'unbiased']:
            _, t, _,  _ = self.run_epoch_and_get_metrics(phase, tracker, epoch, bias_type=bias_type)
            pred_results[bias_type] = t
        value_decomp = calculate_policy_decomposition(pred_results)
        loss_decomp = calculate_outcome_prediction_decomposition(pred_results)
        return {**value_decomp, **loss_decomp}

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def test(self):
        tracker = self.model.tracker
        test_L, test_T, test_metrics, test_res_str = self.run_epoch_and_get_metrics('test', tracker, MAX_EPOCH)
        #do decomposition tracking on test set
        test_decomp_results = self.get_loss_decompositions('test', tracker, MAX_EPOCH)
        
        #do a simple command line print
        msg = 'Test: {}'.format(test_res_str)
        print(msg)
        
        #save results and tensors
        testcsvname = os.path.join(self.results_path, 'test_results.csv')
        self.save_results(test_L, {**test_metrics, **test_decomp_results}, testcsvname, write_headers=True)
        print('Metrics saved to {}'.format(testcsvname))
        self.save_tensors(test_T)
        print('Tensors saved to {}'.format(self.results_path))

    def save_results(self, L, M, csvname, write_headers):
        csv = open(csvname, 'w' if write_headers else 'a')
        # D is a dictionary of metrics: string to float
        k_list = list(L.keys()) + list(M.keys())
        L_and_M = {**L, **M}
        if write_headers:
            s = ','.join(k_list) + '\n'
            csv.write(s)
        s = ','.join(['{:.7f}'.format(L_and_M[k]) for k in k_list]) + '\n'
        csv.write(s)
        csv.close()

    def save_tensors(self, D):
        for k in D:
            fname = os.path.join(self.results_path, '{}.npz'.format(k))
            np.savez(fname, X=D[k])

def get_unbiased_tensor(minibatch, name):
    return np.concatenate([minibatch[name], minibatch['{}_unb'.format(name)]], axis=0)
