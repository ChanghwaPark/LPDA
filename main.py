# Domain Adaptation via Label Propagation

import os
from pprint import pprint

import tensorflow as tf

from models.model import lpda
from trains.train import train

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_string('src', 'svhn', 'Source domain name')
flags.DEFINE_string('trg', 'mnist', 'Target domain name')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_integer('zc', 0, 'Zero centering of data flag')
flags.DEFINE_integer('val', 0, 'Include validation set or not flag')

## Architecture
flags.DEFINE_string('nn', 'lenet', 'Network architecture')
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', '/home/omega/datasets', 'Directory for datasets')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epoch', 80, 'Number of epochs')
flags.DEFINE_integer('inorm', 0, 'Feature extractor instance normalization flag')
flags.DEFINE_integer('trim', 0, 'Which layer to extract feature')
flags.DEFINE_float('dw', 1e-2, 'Adversarial domain adaptation hyper-parameter')
flags.DEFINE_float('lw', 1e-2, 'Label propagation hyper-parameter')
flags.DEFINE_float('wd', 5e-4, 'Weight decay hyper-parameter')
flags.DEFINE_integer('cgw', 1, 'Control gradient weight flag; 0: do not control, 1: do control')
flags.DEFINE_integer('lpc', 1, 'Label propagation closed form flag; 0: iteration, 1: closed form')
flags.DEFINE_integer('lp_iter', 10, 'The number of iterations for label propagation when lpc=0')
flags.DEFINE_integer('adpt', 1, 'Hyper-parameter scheduling for loss_lp and loss_dann flag; 0: no scheduling, 1: yes')
flags.DEFINE_integer('adpt_val', 10, 'Hyper-parameter scheduling speed')

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

    if FLAGS.adpt == 0: FLAGS.adpt_val = 0

    # Define model name
    setup_list = [
        FLAGS.src,
        FLAGS.trg,
        FLAGS.nn,
        f"in_{FLAGS.inorm}",
        f"trim_{FLAGS.trim}",
        f"dw_{FLAGS.dw}",
        f"lw_{FLAGS.lw}",
        f"cgw_{FLAGS.cgw}",
        f"lpc_{FLAGS.lpc}",
        f"adpt_{FLAGS.adpt}_{FLAGS.adpt_val}"
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
