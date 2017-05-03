# -*- coding: utf-8 -*-
"""Parameter initialisers.

The following module defines functions for building a fully connected layer with a non linear output.
"""

import tensorflow as tf

def fc_layer(inputs, input_dim, output_dim):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = tf.matmul(inputs, weights) + biases
    return outputs, weights


def fclayer_regdropout(inputs, input_dim, output_dim, dropout_rate, nonlinearity=tf.nn.relu):
    outputs, weights = fc_layer(inputs, input_dim, output_dim)
    outputs = nonlinearity(tf.nn.dropout(outputs,dropout_rate))
    reg_loss = tf.nn.l2_loss(weights)
    return outputs, reg_loss

