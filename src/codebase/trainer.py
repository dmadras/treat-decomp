import numpy as np
import tensorflow as tf
import os
from codebase.metrics import *
from codebase.utils import make_dir_if_not_exist

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

    def make_feed_dict(self, minibatch):
        return {self.model.X: minibatch['x'], self.model.Y: minibatch['y_f'],
                self.model.Y_cf: minibatch['y_cf'], self.model.T: minibatch['t_f']}

    def process_epoch(self, phase, losses, tensors, epoch):
        epoch_iter = self.data.get_batch_iterator(phase, self.batch_size)
        L = {l: 0. for l in losses}
        T = {t: None for t in tensors}
        self.batches_seen = 0
        for minibatch in epoch_iter:
            self.batches_seen += 1
            feed_dict = self.make_feed_dict(minibatch)
            loss_dict, tensor_dict = self.process_minibatch(phase, feed_dict, losses, tensors)
            L = {k: L[k] + loss_dict[k] for k in L}
            T = {k: np.concatenate((T[k], tensor_dict[k])) if not T[k] is None else tensor_dict[k] for k in T}
        for k in L: L[k] /= self.batches_seen
        return L, T

    def get_metrics(self, tensors, metrics):
        met_dict = {}
        for m in metrics:
            m_fn = metrics[m]
            met_dict[m] = m_fn(tensors)
        return met_dict

    def create_res_str(self, epoch, L, M):
        ep_str = 'E{:d}: '.format(epoch) if not epoch is None else 'Final: '
        res_str = ep_str + ', '.join(['{}:{:.3f}'.format(l, L[l]) for l in L]) \
                        + ', ' + ', '.join(['{}:{:.3f}'.format(m, M[m]) for m in M])
        return res_str

    def run_epoch_and_get_metrics(self, phase, losses, tensors, metrics, epoch):
        L, T = self.process_epoch(phase, losses, tensors, epoch)
        M = self.get_metrics(T, metrics)
        res_str = self.create_res_str(epoch if phase != 'test' else None, L, M)
        return L, T, M, res_str

    def train(self, n_epochs, patience):
        min_val_loss, min_epoch = np.finfo(np.float32).max, -100
        losses = self.model.tracker.losses
        tensors = self.model.tracker.tensors
        metrics = self.model.tracker.metrics
        # losses = {'loss': self.model.loss, 'class_loss': self.model.class_loss}
        # tensors = {'Y_hat': self.model.Y_hat, 'Y': self.model.Y, 'class_loss': self.model.class_loss}
        # metrics = {'errRate': lambda T: errRate(T['Y'], T['Y_hat'])}

        save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
        for epoch in range(n_epochs):
            train_L, train_T, train_metrics, train_res_str = self.run_epoch_and_get_metrics('train', losses, tensors, metrics, epoch)
            valid_L, valid_T, valid_metrics, valid_res_str = self.run_epoch_and_get_metrics('valid', losses, tensors, metrics, epoch)

            #do a simple command line print
            msg = 'Train: {} | Valid: {}'.format(train_res_str, valid_res_str)
            print(msg)

            #do tensorboard tracking (maybe)
            if not self.logs_path is None:
                # Create new Summary objects - move this to external file maybe
                summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)
                summary = tf.Summary()
                for l in losses.keys():
                    summary.value.add(tag=l, simple_value=valid_L[l])
                for m in metrics:
                    summary.value.add(tag=m, simple_value=valid_metrics[m])
                summary_writer.add_summary(summary, epoch)
                summary_writer.flush()

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

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def test(self):
        losses = self.model.tracker.losses
        tensors = self.model.tracker.tensors
        metrics = self.model.tracker.metrics
        # losses = {'loss': self.model.loss, 'class_loss': self.model.class_loss}
        # tensors = {'Y_hat': self.model.Y_hat, 'Y': self.model.Y, 'class_loss': self.model.class_loss}
        # metrics = {'errRate': lambda T: errRate(T['Y'], T['Y_hat'])}
        test_L, test_T, test_metrics, test_res_str = self.run_epoch_and_get_metrics('test', losses, tensors, metrics, MAX_EPOCH)
        
        #do a simple command line print
        msg = 'Test: {}'.format(test_res_str)
        
        #save results and tensors
        self.save_results(test_L, test_metrics)
        self.save_tensors(test_T)
        print(msg)

    def save_results(self, test_L, test_M):
        testcsvname = os.path.join(self.results_path, 'test_results.csv')
        testcsv = open(testcsvname, 'w')
        # D is a dictionary of metrics: string to float
        for D in [test_L, test_M]:
            for k in D:
                s = '{},{:.7f}\n'.format(k, D[k])
                testcsv.write(s)
        testcsv.close()
        print('Metrics saved to {}'.format(testcsvname))

    def save_tensors(self, D):
        for k in D:
            fname = os.path.join(self.results_path, '{}.npz'.format(k))
            np.savez(fname, X=D[k])
        print('Tensors saved to {}'.format(self.results_path))

