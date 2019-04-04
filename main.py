import tensorflow as tf
import numpy as np
import argparse
import functools
import itertools
import operator
import glob
import hooks
from param import Param
from model import Classifier
from dataset import cifar10_input_fn
from network import ResNetWithWeightStandardization

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="cifar10_resnet_model")
parser.add_argument('--filenames', type=str, default="cifar-10-batches-py/data_batch*")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument('--train', action="store_true")
parser.add_argument('--eval', action="store_true")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode, params: Classifier(
            network=ResNetWithWeightStandardization(
                conv_param=Param(filters=32, kernel_size=[3, 3], strides=[1, 1]),
                pool_param=None,
                residual_params=[
                    Param(filters=32, strides=[1, 1], blocks=3),
                    Param(filters=64, strides=[2, 2], blocks=3),
                    Param(filters=128, strides=[2, 2], blocks=3),
                ],
                num_classes=10
            )
        )(features, labels, mode, Param(params)),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=1000
        ),
        params=dict(
            weight_decay=2e-4,
            learning_rate=lambda global_step: tf.train.exponential_decay(
                learning_rate=1e-2,
                global_step=global_step,
                decay_steps=2500000 / args.batch_size,
                decay_rate=0.1
            ),
            momentum=0.9,
            use_nesterov=True
        )
    )

    if args.train:

        estimator.train(
            input_fn=functools.partial(
                cifar10_input_fn,
                filenames=glob.glob(args.filenames),
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                shuffle=True
            )
        )
