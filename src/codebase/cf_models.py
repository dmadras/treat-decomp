from abc import ABC, abstractmethod
import tensorflow as tf
from codebase.mlp import MLP
from codebase.tracker import BinaryCFMLPTracker
from codebase.tf_utils import *

# defaults
HIDDEN_LAYER_SPECS =  {'layer_sizes': [5], 'activ': 'elu'}

class AbstractBaseCFNet(ABC):
    def __init__(self, xdim, ydim, tdim, adim, hidden_layer_specs, seed, **kwargs):
        self.xdim = xdim
        self.ydim = ydim
        self.tdim = tdim
        self.adim = adim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        tf.set_random_seed(self.seed)
        self._define_vars()

    @abstractmethod
    def _define_vars(self): #define tf.placeholders
        pass

    @abstractmethod
    def _get_outcome_logits(self, scope_name='model/preds'): #predict outcome logits from T
        pass

    @abstractmethod
    def _get_outcome_preds(self): #get outcome predictions from logits
        pass

    @abstractmethod
    def _get_outcome_loss(self): #calculate loss on factual outcome predictions
        pass

    @abstractmethod
    def _get_cf_outcome_logits(self, scope_name='model/preds'): #predict CF outcome logits from CF treatments
        pass

    @abstractmethod
    def _get_cf_outcome_preds(self): #get CF outcome predictions from CF logits
        pass

    @abstractmethod
    def _get_cf_outcome_loss(self): #calculate loss on CF outcome predictions
        pass

    @abstractmethod
    def _get_cf_treatments(self): #get CF treatments from treatments
        pass

    @abstractmethod
    def _get_loss(self): #calculate total loss
        pass

class BinaryCFMLP(AbstractBaseCFNet):

    def __init__(self, xdim, ydim, tdim, adim, hidden_layer_specs, seed, **kwargs):

        super().__init__(xdim, ydim, tdim, adim, hidden_layer_specs, seed)
        self.mlp = self._create_outcome_network('outcome_pred', 'model/preds', input_size=self.xdim + self.tdim)
        self.outcome_logits = self._get_outcome_logits()
        self.outcome_preds = self._get_outcome_preds()
        self.outcome_loss = self._get_outcome_loss()
        self.T_cf = self._get_cf_treatments()
        self.cf_outcome_logits = self._get_cf_outcome_logits()
        self.cf_outcome_preds = self._get_cf_outcome_preds()
        self.cf_outcome_loss = self._get_cf_outcome_loss()
        self.loss = self._get_loss()
        self.tracker = BinaryCFMLPTracker(self)

    def _define_vars(self):
        self.X = tf.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.placeholder("float", [None, self.ydim], name='Y')
        self.Y_cf = tf.placeholder("float", [None, self.ydim], name='Y_cf')
        self.T = tf.placeholder("float", [None, self.tdim], name='T')
        self.A = tf.placeholder("float", [None, self.adim], name='A')
        self.epoch = tf.placeholder("float", [1], name='epoch')
        return

    def _get_outcome_logits(self, scope_name='model/preds'):
        return self._get_outcome_logits_from_treatment(self.X, self.T)
    
    def _get_outcome_preds(self):
        return tf.nn.sigmoid(self.outcome_logits)

    def _get_cf_treatments(self):
        return 1. - self.T

    def _get_cf_outcome_logits(self, scope_name='model/preds'):
        return self._get_outcome_logits_from_treatment(self.X, self.T_cf)
    
    def _get_cf_outcome_preds(self):
        return tf.nn.sigmoid(self.cf_outcome_logits)

    def _get_outcome_loss(self):
        return NLL(self.Y, self.outcome_preds)

    def _get_cf_outcome_loss(self):
        return NLL(self.Y_cf, self.cf_outcome_preds)

    def _get_loss(self):
        return tf.reduce_mean(self.outcome_loss) 

    def _create_outcome_network(self, name, scope_name, input_size):
        with tf.variable_scope(scope_name):
            mlp = MLP(name=name,
                      shapes=[input_size] + self.hidden_layer_specs['layer_sizes'] + [self.ydim],
                     activ=self.hidden_layer_specs['activ'])
            return mlp

    def _get_outcome_logits_from_treatment(self, x, t):
        X_and_T = tf.concat([x, t], axis=1)
        logits = self.mlp.forward(X_and_T)
        return logits

class BinaryCFDoubleMLP(BinaryCFMLP):

    def __init__(self, xdim, ydim, tdim, adim, hidden_layer_specs, seed, **kwargs):
        AbstractBaseCFNet.__init__(self, xdim, ydim, tdim, adim, hidden_layer_specs, seed)
        self.mlp_0 = self._create_outcome_network('outcome_pred_0', 'model/preds', input_size=self.xdim)
        self.mlp_1 = self._create_outcome_network('outcome_pred_1', 'model/preds', input_size=self.xdim)
        super().__init__(xdim, ydim, tdim, adim, hidden_layer_specs, seed)

    def _get_outcome_logits_from_treatment(self, x, t):
        logits0 = self.mlp_0.forward(x)
        logits1 = self.mlp_1.forward(x)
        logits = switch(logits0, logits1, t)
        return logits

