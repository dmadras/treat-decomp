from abc import ABC, abstractmethod
import tensorflow as tf
from codebase.mlp import MLP
from codebase.tracker import BinaryMLPTracker
from codebase.tf_utils import *
# defaults
HIDDEN_LAYER_SPECS =  {'layer_sizes': [5], 'activ': 'sigmoid'}

class AbstractBaseNet(ABC):
    def __init__(self, xdim, ydim, hidden_layer_specs, seed, **kwargs):
        self.xdim = xdim
        self.ydim = ydim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        tf.set_random_seed(self.seed)
        self._define_vars()

    @abstractmethod
    def _define_vars(self):
        pass

    @abstractmethod
    def _get_class_logits(self, scope_name='model/preds'): # produce class logits from data  
        pass

    @abstractmethod
    def _get_class_preds(self, scope_name='model/preds'):  # produce class predictions from logits
        pass

    @abstractmethod
    def _get_class_loss(self):  # produce classification loss 
        pass

    @abstractmethod
    def _get_loss(self):  # produce total loss
        pass

class BinaryMLP(AbstractBaseNet):

    def __init__(self, xdim, ydim, hidden_layer_specs, seed, **kwargs):

        super().__init__(xdim, ydim, hidden_layer_specs, seed)
        self.Y_hat_logits = self._get_class_logits()
        self.Y_hat = self._get_class_preds()
        self.class_loss = self._get_class_loss()
        self.loss = self._get_loss()
        self.tracker = BinaryMLPTracker(self)

    def _define_vars(self):
        self.X = tf.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.placeholder("float", [None, self.ydim], name='Y')
        self.epoch = tf.placeholder("float", [1], name='epoch')
        return

    def _get_class_logits(self, scope_name='model/preds'):
        with tf.variable_scope(scope_name):
            mlp = MLP(name='data_to_class_preds',
                      shapes=[self.xdim] + self.hidden_layer_specs['layer_sizes'] + [self.ydim],
                      activ=self.hidden_layer_specs['activ'])
            logits = mlp.forward(self.X)
            return logits

    def _get_class_preds(self, scope_name='model/preds'):
        preds = tf.nn.sigmoid(self.Y_hat_logits)
        return preds

    def _get_class_loss(self):
        return NLL(self.Y, self.Y_hat)

    def _get_loss(self):
        return tf.reduce_mean(self.class_loss) 

