# -*- coding: utf-8 -*-
"""
The following module defines functions for building a fully connected layer with a non linear output.
"""
import tensorflow as tf

def fc_layer(inputs, input_dim, output_dim, dropout_rate, nonlinearity=tf.nn.relu):
	"""Build a fully connected layer.

	Args:
		inputs: array of the inputs (can be the input image, or the output of the previous lyer)
		input_dim: dimension value of input units
		output_dim: dimension value of the output units
		dropout_rate: dropout probability parameter
		nonlinearity: non-linear function applied to the output units

	Returns:
		outputs: outputs unit to be processed by next layer (or make predictions with softmax)
		reg_loss: regulasization penalty error applied to the weights of this layer.

	"""
	weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
	h = tf.matmul(inputs, weights) + biases   
	h_dropout= tf.nn.dropout(h,dropout_rate)
    outputs = nonlinearity(h)
    reg_loss = tf.nn.l2_loss(weights)
    return outputs, reg_loss

