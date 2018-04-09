#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-5 下午5:12
# @Author  : kaixuan
# @Site    : 
# @File    : network.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'
KEEP_PROB = 0.5

def fc_layer(bottom, in_size, out_size, name, bn=True, relu=True,  dropout=True, train_mode=True):
    '''
    set fc_player
    :param bottom:     input layer
    :param in_size:    input size
    :param out_size:   output size
    :param name:       variable_scope name
    :param bn:         batch normal indicate
    :param relu:       relu indicate
    :param sigmoid:    sigmoid indicate
    :param dropout:    dropout indicate
    :param train_mode: trainable indicate
    :return:           fc_player output
    '''
    with tf.variable_scope(name) as scope:
        initial_mode = tf.truncated_normal_initializer(0.0, stddev=0.1)
        weights =make_var('weights', [in_size, out_size], initial_mode, trainable=True)

        initial_mode = tf.constant_initializer(0.1)
        biases = make_var('biases', [out_size], initial_mode, trainable=True)

        fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

        if bn == True:
            fc = batch_norm(fc, is_training=train_mode, is_conv_out=False)

        if relu == True:
            fc = tf.nn.relu(fc)

        # if relu == False and sigmoid == True:
        #     fc = tf.nn.sigmoid(fc)

        if train_mode == True and dropout == True:
            fc = tf.nn.dropout(fc, KEEP_PROB)

        return fc

def conv_layer(bottom, in_channels, out_channels, kernel_size, stride_size, name, padding=DEFAULT_PADDING, bn=True,
                relu=True, train_mode=True):
    '''
    set conv_player
    :param bottom:       input layer
    :param in_channels:  input channel size
    :param out_channels: output channel size
    :param kernel_size:  kernel_size
    :param stride_size:  stride_size
    :param name:         variable_scope name
    :param padding:      padding_mode SAME / VAILD
    :param bn:           batch normal indicate
    :param relu:         relu indicate
    :param train_mode:   trainable indicate
    :return:             conv_player output (with out activation function)
    '''
    with tf.variable_scope(name) as scope:
        initial_mode = tf.truncated_normal_initializer(0.0,stddev=0.1)
        weights = make_var('weights', [kernel_size, kernel_size, in_channels, out_channels], initial_mode, trainable=True)

        initial_mode = tf.constant_initializer(0.0)
        biases = make_var('biases', [out_channels], initial_mode, trainable=True)

        conv = tf.nn.conv2d(bottom, weights, [1, stride_size, stride_size, 1], padding)
        conv = tf.nn.bias_add(conv, biases)

        if bn == True:
            conv = batch_norm(conv, is_training=train_mode, is_conv_out=True)

        if relu == True:
            conv = tf.nn.relu(conv)

        return conv



def max_pool(bottom, kernel_size, stride_size, name, padding=DEFAULT_PADDING):
    '''
    max pool
    :param bottom:      input layer
    :param kernel_size: kernel_size
    :param stride_size: stride_size
    :param name:        name
    :param padding:     padding mode
    :return:            output
    '''
    validate_padding(padding)
    return tf.nn.max_pool(bottom,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding,
                          name=name)


def avg_pool(bottom, kernel_size, stride_size, name, padding=DEFAULT_PADDING):
    '''
    average pool
    :param bottom:      input layer
    :param kernel_size: kernel_size
    :param stride_size: stride_size
    :param name:        name
    :param padding:     padding mode
    :return:            output
    '''
    validate_padding(padding)
    return tf.nn.avg_pool(bottom,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding,
                          name=name)


def make_var(name, shape, initializer=None, trainable=True):
    '''
    product variable
    :param name:        variable name
    :param shape:       variable shape
    :param initializer: initialize mode
    :param trainable:   trainable indicate
    :return:            variable
    '''
    value = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    # self.var_dict[(name, varible[idx])] = var 在此处可保存为npy文件，需要写成类形式，传入biases/weights
    return value


def validate_padding(padding):
    assert padding in ('SAME', 'VALID')


def concat(inputs, axis, name='concat'):
    '''
    concat tensor
    :param inputs: a group of tensor
    :param axis:   dim
    :param name:   name of opera
    :return:
    '''
    return tf.concat(concat_dim=axis, values=inputs, name=name)

def flatten(input):
    '''

    :param input:
    :return:
    '''
    input_shape = input.get_shape().as_list()
    # input_shape = input.shape
    nodes = input_shape[1] * input_shape[2] * input_shape[3]
    flattened_vec = tf.reshape(input, [-1, nodes])
    return flattened_vec


def batch_norm(inputs, is_training, is_conv_out, decay=0.999):
    '''
    copy
    :param inputs:
    :param is_training:
    :param is_conv_out:
    :param decay:
    :return:
    '''
    scale = tf.get_variable('scale', initializer=tf.ones([inputs.get_shape()[-1]]))
    beta = tf.get_variable('beta', initializer=tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.get_variable('pop_mean', initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.get_variable('pop_var', initializer=tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training == True:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)



def load_pretrain_model(data_path, category, session, saver, num_stage):
    '''
    load pretrain model
    :param data_path: model path(ckpt/npy)
    :param session:   session()
    :param saver:     saver
    :param ignore_missing:
    :return:
    '''
    # if data_path.endswith('.ckpt'):
    #     saver.restore(session, data_path)
    if data_path.split('/')[1] == 'model' and data_path.split('/')[2] == category:
        saver.restore(session, data_path)
        print ('Load parameters from ' + data_path)
    else:
        data_dict = np.load(data_path, encoding='latin1').item()
        for stage in range(1,num_stage + 1):
            for key in data_dict:
                with tf.variable_scope('stage_' + str(stage), reuse=True):
                    with tf.variable_scope(key, reuse=True):
                        for subkey in data_dict[key]:
                            try:
                                var = tf.get_variable(subkey)
                                session.run(var.assign(data_dict[key][subkey]))
                                print("assign pretrain model "+subkey+ " to "+ key + ' in stage '+ str(stage))
                                # name = ['weights', 'biases']
                                # var = tf.get_variable(name[subkey])
                                # session.run(var.assign(data_dict[key][subkey]))
                                # print("assign pretrain model "+name[subkey]+ " to "+key)

                            except ValueError:
                                print ("ignore "+key)



def save_npy(self, sess, npy_path="./save.npy"):
    '''
    copy from vgg19.py
    :param self: class
    :param sess: session
    :param npy_path: save path
    :return: save path
    '''
    assert isinstance(sess, tf.Session)

    data_dict = {}

    for (name, idx), var in list(self.var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}

        data_dict[name][idx] = var_out

    np.save(npy_path, data_dict)
    print(("file saved", npy_path))
    return npy_path

