from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def weighed_logistic_loss(scores, labels, pos_loss_mult=1.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    loss_mult = tf.add(tf.mul(labels, pos_loss_mult-neg_loss_mult), neg_loss_mult)

    # Classification loss as the average of weighed per-score loss
    cls_loss = tf.reduce_mean(tf.mul(
        tf.nn.sigmoid_cross_entropy_with_logits(scores, labels),
        loss_mult))

    return cls_loss

def l2_regularization_loss(variables, weight_decay):
    l2_losses = [tf.nn.l2_loss(var) for var in variables]
    total_l2_loss = weight_decay * tf.add_n(l2_losses)
    return total_l2_loss

def smooth_l1_regression_loss(scores, labels, thres=1.0, is_mean=True):
    # L1(x) = 0.5x^2 (|x|<thres)
    # L1(x) = |x|-0.5 (|x|>=thres)
    diff =  tf.abs(scores - labels)
    thres_mat = thres*tf.ones(diff.get_shape())
    # thres_mat = thres*tf.ones((40, 4))
    smooth_sign = tf.cast(tf.less(diff, thres_mat), tf.float32)

    smooth_opt1 = 0.5*tf.multiply(diff, diff)
    smooth_opt2 = diff-0.5

    loss_mat = tf.multiply(smooth_opt1, smooth_sign) + tf.multiply(smooth_opt2, (1.0-smooth_sign))

    if is_mean:
        loss = tf.reduce_mean(loss_mat)
    else:
        loss = loss_mat
    return loss

def ranking_loss(scores, margin=1.0):
    # first column as positive, other columns are negative in scores
    B, num_column = scores.get_shape().as_list()
    num_neg = num_column-1
    loss_vec = tf.zeros(B, tf.float32)
    for i in range(num_neg):
        loss_vec += tf.maximum(0.0, margin-scores[:, 0]+scores[:, 1+i])
    return loss_vec

def kl_divergence(y_dist, x_dist, eps=1e-6): 
    return y_dist * tf.log((y_dist+eps)/(x_dist+eps))