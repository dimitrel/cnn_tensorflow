# -*- coding: utf-8 -*-
"""
The following module defines functions for building a series of convolutional layers.
"""
import tensorflow as tf

def cnn_stack(number_of_layers, channels_out, input_image, channels_in=3, h=32, kernel_size=3):
	"""Build a sequence of convolutional layer in a pyramid structure. That means that the spatial dimensions of each 
	following convolutional layer are reduced by half and its depth is increased. For example, if we use 32x32 RGB images data, 
	the network structure will be the following:
	32x32x3-->16x16xchannels_out-->16x16x(channels_out*2)-->....

	Args:
		number_of_layers : number of convolutional layers for the sequence
		channels_out: output depth dimension of the first convolutional layer (the number of output feature maps)
		input_image: the input image or the output of a previous convolutional layer
		channels_in: input depth dimension (e.g. 3 for an RGB image)
		h: spatial dimensions of the input (e.g. 32 for 32x32 for CIFAR images)
		kernel_size: dimensions of the filter (e.g. 3 for 3x3 filter)

	Returns:
		outputs: output feature maps of a convolutional network (without fc layers)
		output_dimension: dimension of the output feature maps

	"""
	
    inputs=input_image
    for i in range(0,number_of_layers):
        conv_l = conv_layer(inputs,kernel_size,channels_in,channels_out)
        pool_l = max_pool_2x2(conv_l)      
        channels_in=channels_out
        channels_out=channels_out*2
        inputs=pool_l # the next input is the convolutional layer after max pooling
        h=h/2		  # keeping track of spatial dimensions
     
    output_dimension=int(h*h*channels_in)
    output= tf.reshape(pool_l, [-1,output_dimension])
    return output, output_dimension


def conv_layer(input_image, kernel_size, input_dim,output_dim):
	"""Build a convolutional layer with 2xD convolutions.

	Args:
		input_image: the input image or the output of a previous convolutional layer
		input_dim: input depth dimension
		kernel_size: dimensions of the filter (3--> 3x3 filter)
		output_dim: output depth dimension

	Returns:
		outputs: output of the convolutional later

	"""
    weights=tf.Variable(tf.truncated_normal([kernel_size,kernel_size,input_dim,output_dim], stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
    convolutional_layer = tf.nn.conv2d(input_image, weights,strides=[1, 1, 1, 1], padding='SAME')+biases
	outputs = tf.nn.relu(convolutional_layer)
    return outputs


def max_pool_2x2(x):
	"""Perform a 2x2 max pooling operation.

	Args:
		x: the input information (e.g an image)

	Returns: the downsampled input x
		
	"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')