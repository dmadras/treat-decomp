import numpy as np
from codebase.metrics import *


class Tracker(object):

    def __init__(self, model):
        self.model = model
        self.losses = self._get_losses()
        self.tensors = self._get_model_tensors()
        self.base_metrics = self._get_metrics()
        self.metrics = _expand_metrics_by_A(self.base_metrics)
        self.track_tensor_names = self._get_track_tensor_names()

    def _get_losses(self):
        pass

    def _get_model_tensors(self):
        pass

    def _get_metrics(self, metrics):
        pass

    def _get_track_tensor_names(self):
        pass

class BinaryMLPTracker(Tracker):

    def _get_losses(self):
        return {'loss': self.model.loss, 'class_loss': self.model.class_loss}

    def _get_model_tensors(self):
        return {'Y_hat': self.model.Y_hat, 'Y': self.model.Y}

    def _get_metrics(self):
        return {'errRate': lambda T: errRate(T['Y'], T['Y_hat'])}


    def _get_track_tensor_names(self):
        return ['y_cf', 'bayes_f', 'bayes_cf', 'bayes_trmt',  't_prob']        

class BinaryCFMLPTracker(Tracker):

    def _get_losses(self):
        return {'loss': self.model.loss, 'outcome_loss': self.model.outcome_loss, 'cf_outcome_loss': self.model.cf_outcome_loss}

    def _get_model_tensors(self):
        return {'outcome_pred': self.model.outcome_preds, 'outcome': self.model.Y, \
                'cf_outcome_pred': self.model.cf_outcome_preds, 'cf_outcome': self.model.Y_cf, 'treatment': self.model.T,
                'A': self.model.A}

    def _get_metrics(self):
        metrics = {'errRate': lambda T: errRate(T['outcome'], T['outcome_pred']), 
                'cf_errRate': lambda T: errRate(T['cf_outcome'], T['cf_outcome_pred']),
                'PPR': lambda T: PR(T['outcome_pred']), 'cf_PPR': lambda T: PR(T['cf_outcome_pred']),
                'pehe': lambda T: PEHE(T['outcome'], T['cf_outcome'], T['treatment'], T['outcome_pred'], T['cf_outcome_pred']),
                'absErrITE': lambda T: absErrITE(T['outcome'], T['cf_outcome'], T['treatment'], T['outcome_pred'], T['cf_outcome_pred']),
                'absErrATE': lambda T: absErrATE(T['outcome'], T['cf_outcome'], T['treatment'], T['outcome_pred'], T['cf_outcome_pred']),
                'absErrATE-round': lambda T: absErrATE(T['outcome'], T['cf_outcome'], T['treatment'], \
                                        np.round(T['outcome_pred']), np.round(T['cf_outcome_pred'])),
                'AUC': lambda T: AUC(T['outcome'], T['outcome_pred']),
                'cf_AUC': lambda T: AUC(T['cf_outcome'], T['cf_outcome_pred'])
                }
        return metrics

    def _get_track_tensor_names(self):
        return ['y_cf', 'bayes_f', 'bayes_cf', 'bayes_trmt', 't_prob']        


    
def _expand_metrics_by_A(metrics):
    a0_metrics = {'A=0_{}'.format(m): lambda T: metrics[m](_subgroup_dict_items(T, (1 - T['A']))) for m in metrics}
    a1_metrics = {'A=1_{}'.format(m): lambda T: metrics[m](_subgroup_dict_items(T, T['A'])) for m in metrics}
    return {**metrics, **a0_metrics, **a1_metrics}

def _subgroup_dict_items(D, mask):
    Dsub = {k: D[k][mask.astype(bool).flatten()] for k in D}
    return Dsub

