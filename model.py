import tensorflow as tf


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
        loss += tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
        ]) * params.weight_decay

        if mode == tf.estimator.ModeKeys.TRAIN:

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=(
                    params.learning_rate(tf.train.get_or_create_global_step())
                    if callable(params.learning_rate) else params.learning_rate
                ),
                momentum=params.momentum,
                use_nesterov=params.use_nesterov
            )

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_or_create_global_step()
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
