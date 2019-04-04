import tensorflow as tf
import numpy as np
import functools
from ops import *


class ResNetWithBatchNormalization(object):

    def __init__(self, conv_param, pool_param, residual_params, num_classes):

        self.conv_param = conv_param
        self.pool_param = pool_param
        self.residual_params = residual_params
        self.num_classes = num_classes

    def __call__(self, inputs, training=None, name="resnet", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            if self.conv_param:
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=self.conv_param.filters,
                        kernel_size=self.conv_param.kernel_size,
                        strides=self.conv_param.strides,
                        use_bias=False,
                        weight_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros(),
                        apply_weight_standardization=False
                    )

            if self.pool_param:
                inputs = max_pool2d(
                    inputs=inputs,
                    kernel_size=self.pool_param.kernel_size,
                    strides=self.pool_param.strides
                )

            for i, residual_param in enumerate(self.residual_params):

                for j in range(residual_param.blocks)[:1]:
                    with tf.variable_scope("residual_block_{}_{}".format(i, j)):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=residual_param.strides,
                            projection_shortcut=True,
                            training=training
                        )

                for j in range(residual_param.blocks)[1:]:
                    with tf.variable_scope("residual_block_{}_{}".format(i, j)):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=[1, 1],
                            projection_shortcut=False,
                            training=training
                        )

            with tf.variable_scope("norm"):
                inputs = batch_norm(inputs, training=training)

            inputs = tf.nn.relu(inputs)

            inputs = tf.reduce_mean(inputs, axis=[2, 3])

            with tf.variable_scope("logits"):
                inputs = dense(
                    inputs=inputs,
                    units=self.num_classes,
                    use_bias=True,
                    weight_initializer=tf.initializers.glorot_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_weight_standardization=False
                )

            return inputs

    def residual_block(self, inputs, filters, strides, projection_shortcut, training):
        """ A single block for ResNet v2, without a bottleneck.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        (https://arxiv.org/pdf/1603.05027.pdf)
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        """

        shortcut = inputs

        with tf.variable_scope("batch_norm_1st"):
            inputs = batch_norm(inputs, training=training)

        inputs = tf.nn.relu(inputs)

        if projection_shortcut:
            with tf.variable_scope("projection_shortcut"):
                shortcut = conv2d(
                    inputs=inputs,
                    filters=filters,
                    kernel_size=[1, 1],
                    strides=strides,
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_weight_standardization=False
                )

        with tf.variable_scope("conv_1st"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=strides,
                use_bias=False,
                weight_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros(),
                apply_weight_standardization=False
            )

        with tf.variable_scope("batch_norm_2nd"):
            inputs = batch_norm(inputs, training=training)

        inputs = tf.nn.relu(inputs)

        with tf.variable_scope("conv_2nd"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                use_bias=False,
                weight_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros(),
                apply_weight_standardization=False
            )

        inputs += shortcut

        return inputs


class ResNetWithWeightStandardization(object):

    def __init__(self, conv_param, pool_param, residual_params, num_classes):

        self.conv_param = conv_param
        self.pool_param = pool_param
        self.residual_params = residual_params
        self.num_classes = num_classes

    def __call__(self, inputs, training=None, name="resnet", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            if self.conv_param:
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=self.conv_param.filters,
                        kernel_size=self.conv_param.kernel_size,
                        strides=self.conv_param.strides,
                        use_bias=True,
                        weight_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros(),
                        apply_weight_standardization=True
                    )

            if self.pool_param:
                inputs = max_pool2d(
                    inputs=inputs,
                    kernel_size=self.pool_param.kernel_size,
                    strides=self.pool_param.strides
                )

            for i, residual_param in enumerate(self.residual_params):

                for j in range(residual_param.blocks)[:1]:
                    with tf.variable_scope("residual_block_{}_{}".format(i, j)):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=residual_param.strides,
                            projection_shortcut=True,
                            groups=32
                        )

                for j in range(residual_param.blocks)[1:]:
                    with tf.variable_scope("residual_block_{}_{}".format(i, j)):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=[1, 1],
                            projection_shortcut=False,
                            groups=32
                        )

            with tf.variable_scope("norm"):
                inputs = group_norm(inputs, groups=32)

            inputs = tf.nn.relu(inputs)

            inputs = tf.reduce_mean(inputs, axis=[2, 3])

            with tf.variable_scope("logits"):
                inputs = dense(
                    inputs=inputs,
                    units=self.num_classes,
                    use_bias=True,
                    weight_initializer=tf.initializers.glorot_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_weight_standardization=False
                )

            return inputs

    def residual_block(self, inputs, filters, strides, projection_shortcut, groups):
        """ A single block for ResNet v2, without a bottleneck.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        (https://arxiv.org/pdf/1603.05027.pdf)
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        """

        shortcut = inputs

        with tf.variable_scope("group_norm_1st"):
            inputs = group_norm(inputs, groups=groups)

        inputs = tf.nn.relu(inputs)

        if projection_shortcut:
            with tf.variable_scope("projection_shortcut"):
                shortcut = conv2d(
                    inputs=inputs,
                    filters=filters,
                    kernel_size=[1, 1],
                    strides=strides,
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_weight_standardization=True
                )

        with tf.variable_scope("conv_1st"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=strides,
                use_bias=True,
                weight_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros(),
                apply_weight_standardization=True
            )

        with tf.variable_scope("group_norm_2nd"):
            inputs = group_norm(inputs, groups=groups)

        inputs = tf.nn.relu(inputs)

        with tf.variable_scope("conv_2nd"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                use_bias=True,
                weight_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros(),
                apply_weight_standardization=True
            )

        inputs += shortcut

        return inputs
