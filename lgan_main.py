import os
from pprint import pprint

import tensorflow as tf

from models.lgan_model import lgan
from trains.lgan_train import lgan_train

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_string('data', 'mnist', 'LGAN domain name')
flags.DEFINE_integer('sz', 32, 'Experiment input size')
flags.DEFINE_integer('ch', 1, 'Experiment number of channels')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_integer('zc', 0, 'Zero centering of data flag')
flags.DEFINE_integer('val', 0, 'Include validation set or not flag')

## Directory
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', '/home/omega/datasets', 'Directory for datasets')

## Local GAN flags
flags.DEFINE_string('lgan_nn', 'lgan_small', 'LGAN Network architecture')
flags.DEFINE_integer('ngf', 16, 'LGAN generator layers depth')
flags.DEFINE_integer('ndf', 16, 'LGAN discriminator layers depth')
flags.DEFINE_integer('nz', 32, 'Dimension of latent z vector')
flags.DEFINE_integer('jcb', 8, 'Dimension of jacobian')
flags.DEFINE_float('lrD', 5e-5, 'Discriminator learning rate')
flags.DEFINE_float('lrG', 1e-3, 'Generator learning rate')
flags.DEFINE_float('lrDecay', 0.95, 'Decay rate for the learning rate')
flags.DEFINE_float('alpha', 20, 'Weight for locality')
flags.DEFINE_float('beta', 1e-2, 'Weight for orthogonality')
flags.DEFINE_float('theta', 0.1, 'Weight to enforce locality at z = 0')
flags.DEFINE_float('delta', 1e-4, 'Jacobian step size')
flags.DEFINE_float('lgan_var', 3, 'Gaussian noise variance for the training')

## Others
flags.DEFINE_string('gpu', '0', 'GPU number')
flags.DEFINE_string('nn', 'resnet', 'Network architecture for LPDA')

FLAGS = flags.FLAGS


def main(_):
    # Print FLAGS values
    pprint(FLAGS.flag_values_dict())

    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # Define LGAN model name
    lgan_setup_list = [
        FLAGS.lgan_nn,
        FLAGS.data,
        f"{FLAGS.sz}",
        f"{FLAGS.ch}",
        f"ngf_{FLAGS.ngf}",
        f"ndf_{FLAGS.ndf}",
        f"nz_{FLAGS.nz}",
        f"lw_{FLAGS.alpha}",
        f"ow_{FLAGS.beta}",
        f"var_{FLAGS.lgan_var}"
    ]

    lgan_name = '_'.join(lgan_setup_list)
    print(f"LGAN model name: {lgan_name}")

    # Make the models and initialize
    L = lgan(FLAGS, inference=False)
    L.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # Train LGAN model
    lgan_train(L, FLAGS, saver=saver, lgan_name=lgan_name)


if __name__ == '__main__':
    tf.app.run()
