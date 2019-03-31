import tensorflow as tf
import numpy as np


class Classifier(object):

    def __init__(self, network):

        self.network = network

    def __call__(self, images, labels, mode, params):

        logits = self.network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )
        loss += tf.add_n(list(map(tf.nn.l2_loss, tf.trainable_variables()))) * param.weight_decay

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=params.learning_rate,
                    momentum=params.momentum,
                    use_nesterov=True
                )
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_or_create_global_step()
                )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = tf.argmax(
                input=logits,
                axis=-1
            )
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(accuracy=accuracy)
            )
