
import numpy as np
from scipy.stats import norm
from dataproc.utils import sigmoid, save_tensors, generate_normal_data
from codebase.utils import switch


class SynthDatasetCreator(object):

    def __init__(self, seed, num_data, xdim, mu0, mu1, sd0, sd1, p0):
        self.seed = seed 
        np.random.seed(self.seed)
        self.num_data = num_data
        self.xdim = xdim
        self.mu0 = mu0
        self.mu1 = mu1
        self.sd0 = sd0
        self.sd1 = sd1
        self.p0 = p0
        self.p1 = 1. - self.p0
        self.n0 = int(self.num_data * self.p0)
        self.n1 = int(self.num_data * self.p1)

    def generate_dataset(self):
        #make dataset 0
        x0, a0, z0 = self.generate_factual_data(self.mu0, self.sd0, self.n0, self.xdim, 0)    
        x0_cf, a0_cf, z0_cf = self.generate_counterfactual_data(x0, a0, z0, self.mu0, self.sd0, self.mu1, self.sd1)
        dat0 = self.get_treatment_outcome_data(x0, a0, z0) 
        dat0_cf = self.get_treatment_outcome_data(x0_cf, a0_cf, z0_cf) 
        #make dataset 1
        x1, a1, z1 = self.generate_factual_data(self.mu1, self.sd1, self.n1, self.xdim, 1)    
        x1_cf, a1_cf, z1_cf = self.generate_counterfactual_data(x1, a1, z1, self.mu1, self.sd1, self.mu0, self.sd0)
        dat1 = self.get_treatment_outcome_data(x1, a1, z1) 
        dat1_cf = self.get_treatment_outcome_data(x1_cf, a1_cf, z1_cf) 
        #save data
        save_dict_f = {k: np.concatenate([dat0[k], dat1[k]], axis=0) for k in dat0}
        save_dict_cf = {k: np.concatenate([dat0_cf[k], dat1_cf[k]], axis=0) for k in dat0_cf}
        save_dict = {}
        for k in save_dict_f: save_dict[k] = save_dict_f[k]
        for k in save_dict_cf: save_dict['{}_unb'.format(k)] = save_dict_cf[k]
        return save_dict

    def generate_factual_data(self, mu, sd, n, xdim, a_value):
        pass

    def generate_counterfactual_data(self, x, a, z, mu, sd, mu_cf, sd_cf):
        pass

    def get_treatment_outcome_data(self, x, a, z):
        pass

    def calculate_bayes_opt(self, y0_prob, y1_prob, t_obs, t_obs_cf):
        pass

    def choose_observed_treatment(self, x):
        pass

    def generate_outcomes(self, x):
        pass

class BasicSynthDatasetCreator(SynthDatasetCreator):

    def generate_factual_data(self, mu, sd, n, xdim, a_value):
        x = generate_normal_data(mu, sd, n, xdim)
        a = np.full((n, 1), a_value)
        z = np.zeros_like(a)
        print(x.shape, a.shape, z.shape)
        return x, a, z

    def generate_counterfactual_data(self, x, a, z, mu, sd, mu_cf, sd_cf):
        x_base = (x - mu) / sd
        x_cf = (x_base * sd_cf) + mu_cf
        a_cf = 1 - a
        z_cf = z
        return x_cf, a_cf, z_cf

    def get_treatment_outcome_data(self, x, a, z):
        y0, y1, y0_prob, y1_prob = self.generate_outcomes(x)
        t_obs, t_obs_cf, t_obs_prob = self.choose_observed_treatment(x)
        y_factual = np.multiply(y0, 1 - t_obs) + np.multiply(y1, t_obs)
        y_counterfactual = np.multiply(y0, t_obs) + np.multiply(y1, 1 - t_obs)
        bayes_opt_pred_f, bayes_opt_pred_cf = self.calculate_bayes_opt(y0_prob, y1_prob, t_obs, t_obs_cf)
        return {'X': x, 'Y': np.concatenate([y0, y1], axis=1), 'A': a, \
                'T_f': t_obs, 'T_cf': t_obs_cf, 'Y_f': y_factual, 'Y_cf': y_counterfactual, 'Z': z, 
                'bayes_f': bayes_opt_pred_f, 'bayes_cf': bayes_opt_pred_cf, 'T_prob': t_obs_prob}

    def calculate_bayes_opt(self, y0_prob, y1_prob, t_obs, t_obs_cf):
        bayes_opt_pred_t0 = np.round(y0_prob)
        bayes_opt_pred_t1 = np.round(y1_prob)
        bayes_opt_pred_f = switch(bayes_opt_pred_t0, bayes_opt_pred_t1, t_obs)
        bayes_opt_pred_cf = switch(bayes_opt_pred_t0, bayes_opt_pred_t1, t_obs_cf)
        return bayes_opt_pred_f, bayes_opt_pred_cf

    def choose_observed_treatment(self, x):
        t_obs_prob = np.expand_dims(sigmoid(np.sum(x, axis=1) / 5.), 1) 
        t_obs = np.random.binomial(1, t_obs_prob)
        t_obs_cf = 1 - t_obs 
        return t_obs, t_obs_cf, t_obs_prob

    def generate_outcomes(self, x):
        y0_prob = np.expand_dims(sigmoid(np.sum(x, axis=1) / 5.), 1)
        y1_prob = np.expand_dims(sigmoid(-np.sum(x, axis=1) / 5.), 1)
        y0 = np.random.binomial(1, y0_prob)
        y1 = np.random.binomial(1, y1_prob)
        return y0, y1, y0_prob, y1_prob 



