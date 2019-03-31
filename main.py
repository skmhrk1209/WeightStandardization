import tensorflow as tf
import numpy as np
import argparse
import functools
import glob
import hooks
from param import Param
from dataset import cifar10_input_fn
from model import Classifier
from network import ResNet

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="cifar10_resnet_model")
parser.add_argument('--train_filenames', type=str, default="cifar-10-batches-py/data_batch*")
parser.add_argument('--valid_filenames', type=str, default="cifar-10-batches-py/test_batch*")
parser.add_argument('--test_filenames', type=str, default="cifar-10-batches-py/test_batch*")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument('--train', action="store_true")
parser.add_argument('--eval', action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode, params: Classifier(
            network=ResNet(
                conv_param=Param(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=Param(kernel_size=[3, 3], strides=[2, 2]),
                residual_params=[
                    Param(filters=64, strides=[2, 2], blocks=2),
                    Param(filters=128, strides=[2, 2], blocks=2),
                    Param(filters=256, strides=[2, 2], blocks=2),
                    Param(filters=512, strides=[2, 2], blocks=2),
                ],
                num_classes=10,
                apply_weight_standardization=True
            )
        )(features, labels, mode, Param(params)),
        params=dict(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
        ),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        estimator.train(
            input_fn=functools.partial(
                cifar10_input_fn,
                filenames=glob.glob(args.train_filenames),
                batch_size=args.batch_size,
                num_epochs=args.num_epochs if args.train else 1,
                shuffle=True if args.train else False,
            ),
            max_steps=args.max_steps,
            hooks=[
                hooks.ValidationMonitorHook(
                    estimator=estimator,
                    input_fn=functools.partial(
                        cifar10_input_fn,
                        filenames=glob.glob(args.valid_filenames),
                        batch_size=args.batch_size,
                        num_epochs=1,
                        shuffle=False,
                    ),
                    every_n_steps=1000,
                    steps=1000,
                    name="validation"
                )
            ]
        )

    if args.eval:

        eval_result = estimator.evaluate(
            input_fn=functools.partial(
                cifar10_input_fn,
                filenames=glob.glob(args.test_filenames),
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False,
            ),
            steps=args.steps,
            name="test"
        )

        print("==================================================")
        tf.logging.info("test result")
        tf.logging.info(eval_result)
        print("==================================================")
