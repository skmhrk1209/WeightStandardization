import tensorflow as tf
import numpy as np
import argparse
import functools
import itertools
import operator
import glob
import hooks
from param import Param
from dataset import cifar10_input_fn
from network import ResNetWithWeightStandardization

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="cifar10_resnet_model")
parser.add_argument('--filenames', type=str, default="cifar-10-batches-py/data_batch*")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument('--train', action="store_true")
parser.add_argument('--eval', action="store_true")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    classifier = ResNetWithWeightStandardization(
        conv_param=Param(filters=32, kernel_size=[3, 3], strides=[1, 1]),
        pool_param=None,
        residual_params=[
            Param(filters=32, strides=[1, 1], blocks=3),
            Param(filters=64, strides=[2, 2], blocks=3),
            Param(filters=128, strides=[2, 2], blocks=3),
        ],
        num_classes=10,
    )

    images, labels = cifar10_input_fn(
        filenames=glob.glob(args.filenames),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs if args.train else 1,
        shuffle=True if args.train else False,
    )

    if args.train:

        logits = classifier(images)
        losses = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
            reduction=tf.losses.Reduction.NONE
        )
        losses += tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
        ]) * 2e-4

        global_step = tf.train.create_global_step()
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=tf.train.piecewise_constant(
                x=global_step,
                boundaries=[50000 * num_epochs // args.batch_size for num_epochs in [100, 150, 200]],
                values=[0.1 * decay_rate for decay_rate in [1.0, 0.1, 0.01, 0.001]]
            ),
            momentum=0.9,
            use_nesterov=True
        )

        def generator():
            global variables
            for losses in tf.split(losses, num_or_size_splits=args.batch_size, axis=0):
                loss = tf.reduce_mean(losses)
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                yield gradients

        loss = tf.reduce_mean(losses)
        gradients = map(functools.partial(tf.reduce_mean, axis=0), zip(*generator()))

        train_op = optimizer.apply_gradients(
            grads_and_vars=zip(gradients, variables),
            global_step=global_step
        )

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=args.model_dir,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=args.model_dir,
                    save_steps=1000,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    )
                ),
                tf.train.SummarySaverHook(
                    output_dir=args.model_dir,
                    save_steps=100,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar("loss", loss)
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=global_step,
                        loss=loss
                    ),
                    every_n_iter=100,
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(train_op)
