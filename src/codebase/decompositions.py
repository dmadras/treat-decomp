import numpy as np
from codebase.metrics import *


'''Want to be able to compute decomposition terms from epoch results.
Input: a dictionary with results on the test set for normal, flipped, and unbiased data sets.
Compute: V, V_sample, V_treat, V*, treatment shift.
Maybe:also noise (same as V*), bias (need to average over datasets), variance (need to average over datasets)?'''

def choose_treatment(pred0, pred1):
    t0 = np.greater(pred0, pred1)
    t1 = np.less(pred0, pred1)
    tied = np.equal(pred0, pred1)
    tiebroken = np.multiply(tied, np.random.binomial(1, 0.5, size=tied.shape))
    chosen_t = 0. * t0 + 1. * t1 + tiebroken
    return chosen_t

def calculate_value(t, v0, v1):
    return np.mean(switch(v0, v1, t))

def calculate_policy_values(D):
    #V, V_sample, V_treat, V*
    #calculate value of learned policy on observed sample
    trmt = choose_treatment(D['normal']['cf_outcome_pred'], D['normal']['outcome_pred'])
    V = calculate_value(trmt, D['normal']['cf_outcome'], D['normal']['outcome'])
    #calculate value of learned policy on unbiased sample
    trmt_sample = choose_treatment(D['unbiased']['cf_outcome_pred'], D['unbiased']['outcome_pred'])
    V_sample = calculate_value(trmt_sample, D['unbiased']['cf_outcome'], D['unbiased']['outcome'])
    #calculate value of optimal policy on observed sample
    trmt_treat = choose_treatment(D['normal']['bayes_cf'], D['normal']['bayes_f'])
    V_treat = calculate_value(trmt_treat, D['normal']['cf_outcome'], D['normal']['outcome'])
    #calculate value of optimal policy on unbiased sample
    trmt_star = choose_treatment(D['unbiased']['bayes_cf'], D['unbiased']['bayes_f'])
    V_star = calculate_value(trmt_star, D['unbiased']['cf_outcome'], D['unbiased']['outcome'])
    return {'V': V,
            'V_sample': V_sample,
            'V_treat': V_treat,
            'V_star': V_star}

def calculate_outcome_prediction_decomposition(D):
    #L, L_do, treatment_shift: L_do = L + treatment_shift
    #L
    err_f = avg_ce(D['normal']['outcome'], D['normal']['outcome_pred'])
    L = err_f
    #L_do
    err_cf = avg_ce(D['normal']['cf_outcome'], D['normal']['cf_outcome_pred'])
    L_do = 0.5 * (err_f + err_cf)
    #treatment shift term
    t_prob = D['normal']['t_prob']
    prob_t1 = np.greater(t_prob, 0.5)
    trmt_likely = np.equal(D['normal']['treatment'], prob_t1.astype(float))
    likely_trmt_outcomes = switch(D['normal']['cf_outcome'], D['normal']['outcome'], trmt_likely)
    unlikely_trmt_outcomes = switch(D['normal']['cf_outcome'], D['normal']['outcome'], 1 - trmt_likely)
    likely_trmt_preds = switch(D['normal']['cf_outcome_pred'], D['normal']['outcome_pred'], trmt_likely)
    unlikely_trmt_preds = switch(D['normal']['cf_outcome_pred'], D['normal']['outcome_pred'], 1 - trmt_likely)
    L_likely_trmts = ce(likely_trmt_outcomes, likely_trmt_preds)
    L_unlikely_trmts = ce(unlikely_trmt_outcomes, unlikely_trmt_preds)
    delta_t_prob = np.abs(t_prob - 0.5)
    treatment_shift_loss = np.mean(np.multiply(delta_t_prob, L_unlikely_trmts - L_likely_trmts))
    return {'L': L, 'L_do': L_do, 't_shift': treatment_shift_loss}

    

