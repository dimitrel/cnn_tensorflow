# -*- coding: utf-8 -*-
"""Parameter initialisers.

The following module defines functions for building a series of convolutional layers.
"""

import tensorflow as tf

def cnn_stack(number_of_layers,channels_out,input_image,channels_in=3,h=32,kernel_size=3):
    
    inputs=input_image
    for i in range(0,number_of_layers):
        conv_l = conv_layer(inputs,kernel_size,channels_in,channels_out)
        pool_l = max_pool_2x2(conv_l)      
        channels_in=channels_out
        channels_out=channels_out*2
        inputs=pool_l
        h=h/2
     
    output_dimension=int(h*h*channels_in)
    output= tf.reshape(pool_l, [-1,output_dimension])
    return output, output_dimension


def conv_layer(input_image,kernel_size,input_dim,output_dim):
    weights=tf.Variable(tf.truncated_normal([kernel_size,kernel_size,input_dim,output_dim], stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
    conv_out = tf.nn.conv2d(input_image, weights,strides=[1, 1, 1, 1], padding='SAME')+biases
    return tf.nn.relu(conv_out)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')