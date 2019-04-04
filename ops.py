import tensorflow as tf
import numpy as np


def weight_standardization(weight, epsilon=1e-8):
    shape = weight.shape.as_list()
    weight = tf.reshape(weight, [-1, shape[-1]])
    weight -= tf.math.reduce_mean(weight, axis=0, keepdims=True)
    weight /= tf.math.reduce_std(tf.square(weight), axis=0, keepdims=True)
    weight = tf.reshape(weight, shape)
    return weight


def group_norm(inputs, groups):
    return tf.contrib.layers.group_norm(
        inputs=inputs,
        groups=groups,
        channels_axis=1,
        reduction_axes=[2, 3],
        center=True,
        scale=True
    )


def batch_norm(inputs, training):
    return tf.contrib.layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=training,
        data_format="NCHW"
    )


def get_weight(shape, initializer, apply_weight_standardization=False):
    weight = tf.get_variable(
        name="weight",
        shape=shape,
        initializer=initializer
    )
    if apply_weight_standardization:
        weight = weight_standardization(weight)
    return weight


def get_bias(shape, initializer):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=initializer
    )
    return bias


def dense(inputs, units, use_bias=True,
          weight_initializer=tf.initializers.he_normal(),
          bias_initializer=tf.initializers.zeros(),
          apply_weight_standardization=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        initializer=weight_initializer,
        apply_weight_standardization=apply_weight_standardization
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, filters, kernel_size,
           strides=[1, 1], use_bias=True,
           weight_initializer=tf.initializers.he_normal(),
           bias_initializer=tf.initializers.zeros(),
           apply_weight_standardization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        initializer=weight_initializer,
        apply_weight_standardization=apply_weight_standardization
    )
    inputs = tf.nn.conv2d(
        input=inputs,
        filter=weight,
        strides=[1, 1] + strides,
        padding="SAME",
        data_format="NCHW"
    )
    if use_bias:
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def conv2d_transpose(inputs, filters, kernel_size,
                     strides=[1, 1], use_bias=True,
                     weight_initializer=tf.initializers.he_normal(),
                     bias_initializer=tf.initializers.zeros(),
                     apply_weight_standardization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        initializer=weight_initializer,
        apply_weight_standardization=apply_weight_standardization
    )
    weight = tf.transpose(weight, [0, 1, 3, 2])
    input_shape = np.array(inputs.shape.as_list())
    output_shape = [input_shape[0], filters, *input_shape[2:] * strides]
    inputs = tf.nn.conv2d_transpose(
        value=inputs,
        filter=weight,
        output_shape=output_shape,
        strides=[1, 1] + strides,
        padding="SAME",
        data_format="NCHW"
    )
    if use_bias:
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def max_pool2d(inputs, kernel_size, strides):
    return tf.contrib.layers.max_pool2d(
        inputs=inputs,
        kernel_size=kernel_size,
        stride=strides,
        padding="SAME",
        data_format="NCHW"
    )
