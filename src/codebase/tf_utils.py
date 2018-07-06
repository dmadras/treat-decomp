import tensorflow as tf
import numpy as np

EPS = 1e-12


def NLL(target, pred, eps=EPS):
    l = -(tf.multiply(target, tf.log(pred + eps)) + tf.multiply(1 - target, tf.log(1 - pred + eps)))
    return l

def classification_error(target, pred):
    pred_class = tf.round(pred)
    l = 1.0 - tf.reduce_mean(tf.cast(tf.equal(target, pred_class), tf.float32))
    return l

def soft_switch(x0, x1, s):
    return tf.multiply(x0, 1. - s) + tf.multiply(x1, s)

def switch(x0, x1, s):
    s_ind = tf.cast(tf.greater(s, 0.5), tf.float32)
    return tf.multiply(x0, 1. - s_ind) + tf.multiply(x1, s_ind)

