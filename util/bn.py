from __future__ import absolute_import, division, print_function

import tensorflow as tf

def batch_norm(inputs, is_training, scope, decay=0.999, epsilon=1e-3):

    with tf.variable_scope(scope):
        scale_init = tf.ones([inputs.get_shape()[-1]])
        scale = tf.get_variable('scale', initializer=scale_init)
        beta_init = tf.zeros([inputs.get_shape()[-1]])
        beta = tf.get_variable('beta', initializer=beta_init)
        pop_mean_init = tf.zeros([inputs.get_shape()[-1]])
        pop_mean = tf.get_variable('pop_mean', initializer=pop_mean_init, trainable=False)
        pop_var_init = tf.ones([inputs.get_shape()[-1]])
        pop_var = tf.get_variable('pop_var', initializer=pop_var_init, trainable=False)
        # print(pop_mean)

        def training_pop_static():
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        def evaluating_pop_static():
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)

        result = tf.cond(is_training,
                        training_pop_static,
                        evaluating_pop_static)
        return result