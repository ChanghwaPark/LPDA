# Domain Adaptation via Label Propagation

import os
from pprint import pprint

import tensorflow as tf

from models.model import lpda
from trains.train import train
from data.dataset import get_attr

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_string('src', 'svhn', 'Source domain name')
flags.DEFINE_string('trg', 'mnist', 'Target domain name')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_integer('zc', 0, 'Zero centering of data flag')
flags.DEFINE_integer('val', 0, 'Include validation set or not flag')

## Architecture
flags.DEFINE_string('nn', 'small_lenet', 'Network architecture')
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', '/home/omega/datasets', 'Directory for datasets')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epoch', 80, 'Number of epochs')
flags.DEFINE_integer('inorm', 0, 'Feature extractor instance normalization flag')
flags.DEFINE_integer('trim', 3, 'Which layer to extract feature')
flags.DEFINE_float('dw', 1e-2, 'Adversarial domain adaptation hyper-parameter')

## Others
flags.DEFINE_string('gpu', '0', 'GPU number')

FLAGS = flags.FLAGS


def main(_):
    # Print FLAGS values
    pprint(FLAGS.flag_values_dict())

    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # Define model name
    setup_list = [
        FLAGS.src,
        FLAGS.trg,
        FLAGS.nn,
        f"in_{FLAGS.inorm}",
        f"trim_{FLAGS.trim}",
        f"dw_{FLAGS.dw}"
    ]
    model_name = '_'.join(setup_list)
    print(f"Model name: {model_name}")

    # Make main model and initialize
    M = lpda(FLAGS)
    M.sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # Train the main model
    train(M, FLAGS, saver=saver, model_name=model_name)


if __name__ == '__main__':
    tf.app.run()
