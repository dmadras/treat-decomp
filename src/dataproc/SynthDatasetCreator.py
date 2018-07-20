
import numpy as np
from scipy.stats import norm
from dataproc.utils import sigmoid, save_tensors, generate_normal_data, generate_bernoulli_data
from codebase.utils import switch


class SynthDatasetCreator(object):

    def __init__(self, seed, num_data, xdim, mu0, mu1, sd0, sd1, p0, outcome_scale):
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
        self.outcome_scale = outcome_scale

    def generate_dataset(self):
        save_dict_f, save_dict_cf = self.generate_factual_and_counterfactual_data()
        save_dict = {}
        for k in save_dict_f: save_dict[k] = save_dict_f[k]
        for k in save_dict_cf: save_dict['{}_unb'.format(k)] = save_dict_cf[k]
        return save_dict

    def generate_factual_and_counterfactual_data(self):
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
        return save_dict_f, save_dict_cf

    def generate_factual_data(self, mu, sd, n, xdim, a_value):
        pass

    def generate_counterfactual_data(self, x, a, z, mu, sd, mu_cf, sd_cf):
        pass

    def get_treatment_outcome_data(self, x, a, z):
        pass

    def calculate_bayes_opt(self, y0_prob, y1_prob, t_obs, t_obs_cf):
        pass

    def choose_observed_treatment(self, x, a, z):
        pass

    def generate_outcomes(self, x, a, z):
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
        y0, y1, y0_prob, y1_prob = self.generate_outcomes(x, a, z)
        t_obs, t_obs_cf, t_obs_prob = self.choose_observed_treatment(x, a, z)
        y_factual = np.multiply(y0, 1 - t_obs) + np.multiply(y1, t_obs)
        y_counterfactual = np.multiply(y0, t_obs) + np.multiply(y1, 1 - t_obs)
        bayes_opt_pred_f, bayes_opt_pred_cf = self.calculate_bayes_opt(y0_prob, y1_prob, t_obs, t_obs_cf)
        return {'X': x, 'Y': np.concatenate([y0, y1], axis=1), 'A': a, 'y0': y0, 'y1': y1, \
                'T_f': t_obs, 'T_cf': t_obs_cf, 'Y_f': y_factual, 'Y_cf': y_counterfactual, 'Z': z, 
                'bayes_f': bayes_opt_pred_f, 'bayes_cf': bayes_opt_pred_cf, 'T_prob': t_obs_prob}

    def calculate_bayes_opt(self, y0_prob, y1_prob, t_obs, t_obs_cf):
        bayes_opt_pred_t0 = np.round(y0_prob)
        bayes_opt_pred_t1 = np.round(y1_prob)
        bayes_opt_pred_f = switch(bayes_opt_pred_t0, bayes_opt_pred_t1, t_obs)
        bayes_opt_pred_cf = switch(bayes_opt_pred_t0, bayes_opt_pred_t1, t_obs_cf)
        return bayes_opt_pred_f, bayes_opt_pred_cf

    def choose_observed_treatment(self, x, a, z):
        t_obs_prob = np.expand_dims(sigmoid(np.mean(x, axis=1) / self.outcome_scale), 1) 
        t_obs = np.random.binomial(1, t_obs_prob)
        t_obs_cf = 1 - t_obs 
        return t_obs, t_obs_cf, t_obs_prob

    def generate_outcomes(self, x, a, z):
        y0_prob = np.expand_dims(sigmoid(np.mean(x, axis=1) / self.outcome_scale), 1)
        y1_prob = np.expand_dims(sigmoid(-np.mean(x, axis=1) / self.outcome_scale), 1)
        y0 = np.random.binomial(1, y0_prob)
        y1 = np.random.binomial(1, y1_prob)
        return y0, y1, y0_prob, y1_prob 


class OffsetOutcomeSynthDatasetCreator(BasicSynthDatasetCreator):

    def __init__(self, seed, offset, **kwargs): 
        self.offset = offset
        super().__init__(seed, **kwargs)

    def generate_outcomes(self, x, a, z):
        mn = np.mean(x, axis=1)
        logit_below_0 = mn / self.outcome_scale
        logit_above_0 = (mn + self.offset) / self.outcome_scale
        logits = switch(logit_below_0, logit_above_0, mn > 0)
        y0_prob = np.expand_dims(sigmoid(logits), 1)
        y1_prob = np.expand_dims(sigmoid(-logits), 1)
        y0 = np.random.binomial(1, y0_prob)
        y1 = np.random.binomial(1, y1_prob)
        return y0, y1, y0_prob, y1_prob 

class ConfoundedTreatmentBasicSynthDatasetCreator(BasicSynthDatasetCreator):

    def __init__(self, seed, treatment_offset, **kwargs): 
        self.treatment_offset = treatment_offset
        super().__init__(seed, **kwargs)

    def choose_observed_treatment(self, x, a, z):
        # A = 0 treatment
        a0_t_obs_prob = np.expand_dims(sigmoid(np.mean(x, axis=1) / self.outcome_scale), 1) 
        a0_t_obs = np.random.binomial(1, a0_t_obs_prob)
        a0_t_obs_cf = 1 - a0_t_obs 
        # A = 1 treatment
        a1_t_obs_prob = np.expand_dims(sigmoid(np.mean(x + self.treatment_offset, axis=1) / self.outcome_scale), 1) 
        a1_t_obs = np.random.binomial(1, a1_t_obs_prob)
        a1_t_obs_cf = 1 - a1_t_obs 
        #combine outcomes by A
        t_obs = switch(a0_t_obs, a1_t_obs, a)
        t_obs_prob = switch(a0_t_obs_prob, a1_t_obs_prob, a)
        t_obs_cf = switch(a0_t_obs_cf, a1_t_obs_cf, a)
        return t_obs, t_obs_cf, t_obs_prob


class ConfoundedSynthDatasetCreator(ConfoundedTreatmentBasicSynthDatasetCreator):

    def __init__(self, seed, offset, **kwargs): 
        self.offset = offset
        super().__init__(seed, **kwargs)

    def generate_outcomes(self, x, a, z):
        #A = 0 outcomes
        a0_y0_prob = np.expand_dims(sigmoid(np.mean(x, axis=1) / self.outcome_scale), 1)
        a0_y1_prob = np.expand_dims(sigmoid(-np.mean(x, axis=1) / self.outcome_scale), 1)
        a0_y0 = np.random.binomial(1, a0_y0_prob)
        a0_y1 = np.random.binomial(1, a0_y1_prob)
        #A = 1 outcomes
        a1_y0_prob = np.expand_dims(sigmoid(np.mean(x + self.offset, axis=1) / self.outcome_scale), 1)
        a1_y1_prob = np.expand_dims(sigmoid(-np.mean(x + self.offset, axis=1) / self.outcome_scale), 1)
        a1_y0 = np.random.binomial(1, a1_y0_prob)
        a1_y1 = np.random.binomial(1, a1_y1_prob)
        #combine outcomes by A
        y0 = switch(a0_y0, a1_y0, a)
        y1 = switch(a0_y1, a1_y1, a)
        y0_prob = switch(a0_y0_prob, a1_y0_prob, a)
        y1_prob = switch(a0_y1_prob, a1_y1_prob, a)
        return y0, y1, y0_prob, y1_prob 

class HiddenConfounderSynthDatasetCreator(BasicSynthDatasetCreator):

    def __init__(self, seed, zdim, zp, **kwargs): 
        self.zdim = zdim
        self.zp = zp
        super().__init__(seed, **kwargs)

    def generate_factual_data(self, mu, sd, n, xdim, a_value):
        a = np.full((n, 1), a_value)
        z = generate_bernoulli_data(self.zp, n, self.zdim)
        x = generate_normal_data(z + mu, sd, n, self.xdim)
        print(x.shape, a.shape, z.shape)
        return x, a, z

    def generate_counterfactual_data(self, x, a, z, mu, sd, mu_cf, sd_cf):
        x_base = (x - mu - z) / sd
        x_cf = (x_base * sd_cf) + mu_cf + z
        a_cf = 1 - a
        z_cf = z
        return x_cf, a_cf, z_cf

    def choose_observed_treatment(self, x, a, z):
        # A = 0 treatment
        a0_t_obs_prob = np.expand_dims(sigmoid(np.mean(x + z, axis=1) / self.outcome_scale), 1) 
        a0_t_obs = np.random.binomial(1, a0_t_obs_prob)
        a0_t_obs_cf = 1 - a0_t_obs 
        # A = 1 treatment
        a1_t_obs_prob = np.expand_dims(sigmoid(np.mean(x - z, axis=1) / self.outcome_scale), 1) 
        a1_t_obs = np.random.binomial(1, a1_t_obs_prob)
        a1_t_obs_cf = 1 - a1_t_obs 
        #combine outcomes by A
        t_obs = switch(a0_t_obs, a1_t_obs, a)
        t_obs_prob = switch(a0_t_obs_prob, a1_t_obs_prob, a)
        t_obs_cf = switch(a0_t_obs_cf, a1_t_obs_cf, a)
        return t_obs, t_obs_cf, t_obs_prob

    def generate_outcomes(self, x, a, z):
        #A = 0 outcomes
        a0_y0_prob = np.expand_dims(sigmoid(np.mean(x + z, axis=1) / self.outcome_scale), 1)
        a0_y1_prob = np.expand_dims(sigmoid(-np.mean(x - z, axis=1) / self.outcome_scale), 1)
        a0_y0 = np.random.binomial(1, a0_y0_prob)
        a0_y1 = np.random.binomial(1, a0_y1_prob)
        #A = 1 outcomes
        a1_y0_prob = np.expand_dims(sigmoid(np.mean(x - z, axis=1) / self.outcome_scale), 1)
        a1_y1_prob = np.expand_dims(sigmoid(-np.mean(x + z, axis=1) / self.outcome_scale), 1)
        a1_y0 = np.random.binomial(1, a1_y0_prob)
        a1_y1 = np.random.binomial(1, a1_y1_prob)
        #combine outcomes by A
        y0 = switch(a0_y0, a1_y0, a)
        y1 = switch(a0_y1, a1_y1, a)
        y0_prob = switch(a0_y0_prob, a1_y0_prob, a)
        y1_prob = switch(a0_y1_prob, a1_y1_prob, a)
        return y0, y1, y0_prob, y1_prob 


class CorrAZSynthDatasetCreator(HiddenConfounderSynthDatasetCreator):

    def __init__(self, seed, beta_lo, beta_hi, **kwargs): 
        self.beta_lo = beta_lo
        self.beta_hi = beta_hi
        super().__init__(seed, **kwargs)

    def generate_factual_and_counterfactual_data(self):
        #everything the same, but then select observed A at the end
        #make dataset
        x0, a0, z0 = self.generate_factual_data(self.mu0, self.sd0, self.num_data, self.xdim, 0)    
        x0_cf, a0_cf, z0_cf = self.generate_counterfactual_data(x0, a0, z0, self.mu0, self.sd0, self.mu1, self.sd1)
        dat0 = self.get_treatment_outcome_data(x0, a0, z0) 
        dat0_cf = self.get_treatment_outcome_data(x0_cf, a0_cf, z0_cf) 
        #select observed sensitive attribute as a function of Z
        observed_a = self.choose_observed_sensitive_attribute(z0)
        #save data
        save_dict_f = {k: switch(dat0[k], dat0_cf[k], observed_a) for k in dat0}
        save_dict_cf = {k: switch(dat0[k], dat0_cf[k], 1 - observed_a) for k in dat0}
        return save_dict_f, save_dict_cf

    def choose_observed_sensitive_attribute(self, z):
        zval = np.expand_dims(np.mean(z, axis=1) > 0.5, 1)
        alpha = self.beta_lo * (1 - zval) + self.beta_hi * zval
        beta = self.beta_hi * (1 - zval) + self.beta_lo * zval
        bern_p = np.random.beta(alpha, beta)
        cutoff = np.percentile(bern_p, self.p0 * 100)
        observed_a = bern_p > cutoff
        return observed_a.astype(float)

