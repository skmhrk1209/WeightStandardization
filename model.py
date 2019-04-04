import tensorflow as tf


class Classifier(object):

    def __init__(self, network):

        self.network = network

    def __call__(self, images, labels, mode, params):

        logits = self.network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        losses = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
            reduction=tf.losses.Reduction.NONE
        )
        losses += tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
        ]) * params.weight_decay

        loss = tf.reduce_mean(losses)

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

                def generator(losses, iterations):
                    global variables
                    for losses in tf.split(losses, iterations, axis=0):
                        loss = tf.reduce_mean(losses)
                        gradients, variables = zip(*optimizer.compute_gradients(loss))
                        yield gradients

                gradients = [
                    tf.reduce_mean(gradients, axis=0)
                    for gradients in zip(*generator(losses, params.iterations))
                ]

                train_op = optimizer.apply_gradients(
                    grads_and_vars=zip(gradients, variables),
                    global_step=tf.train.create_global_step()
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
