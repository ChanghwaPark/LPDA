# Domain Adaptation via Label Propagation

import os
from pprint import pprint

import tensorflow as tf
from termcolor import colored

from data.dataset import get_attr, get_attr_image
from models.lgan_model import lgan
from models.model import lpda
from models.model_image import lpda_image
from trains.train import train
from trains.train_image import train_image
from utils import get_lgan_name, update_lgan_flags

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_string('src', 'mnist', 'Source domain name')
flags.DEFINE_string('trg', 'svhn', 'Target domain name')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_integer('zc', 0, 'Zero centering of data flag')
flags.DEFINE_integer('val', 0, 'Include validation set or not flag')
flags.DEFINE_boolean('src_inv', False, 'Randomly invert source images flag')
flags.DEFINE_boolean('trg_inv', False, 'Randomly invert target images flag')

## Architecture
flags.DEFINE_string('nn', 'small', 'Network architecture')
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', '/home/omega/datasets', 'Directory for datasets')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epoch', 80, 'Number of epochs')
flags.DEFINE_integer('inorm', 0, 'Feature extractor instance normalization flag')
flags.DEFINE_integer('trim', 0, 'Which layer to extract feature')
flags.DEFINE_float('dw', 1e-1, 'Adversarial domain adaptation hyper-parameter')
flags.DEFINE_float('lw', 1, 'Label propagation hyper-parameter')
flags.DEFINE_float('wd', 5e-4, 'Weight decay hyper-parameter')
flags.DEFINE_integer('cgw', 1, 'Control gradient weight flag; 0: do not control, 1: do control')
flags.DEFINE_integer('lpc', 0, 'Label propagation closed form flag; 0: iteration, 1: closed form')
flags.DEFINE_integer('lp_iter', 10, 'The number of iterations for label propagation when lpc=0')
flags.DEFINE_integer('adpt', 1, 'Hyper-parameter scheduling for loss_lp and loss_dann flag; 0: no scheduling, 1: yes')
flags.DEFINE_integer('adpt_val', 10, 'Hyper-parameter scheduling speed')
flags.DEFINE_float('grad_val', 1, 'Gradient weight control parameter')
flags.DEFINE_float('xw', 0.1, 'Target prediction cross entropy loss weight')
flags.DEFINE_float('sw', 0.1, 'LGAN regularization on source domain weight')
flags.DEFINE_float('tw', 0.1, 'LGAN regularization on target domain weight')
flags.DEFINE_integer('pn', 0, 'Regularizer perturbation normalization flag; 0: do not normalize z, 1: normalize z')
flags.DEFINE_float('sv', 0.1, 'Source mixing policy std for gaussian; alpha for beta function; std for LGAN loss')
flags.DEFINE_float('tv', 0.1, 'Target mixing policy std for gaussian; alpha for beta function; std for LGAN loss')
flags.DEFINE_float('vsw', 0, 'Source VAT loss weight')
flags.DEFINE_float('vtw', 0, 'Target VAT loss weight')
flags.DEFINE_string('lp_loss', 'l1', 'Cycle loss term loss function; l1, l2, ce')
flags.DEFINE_float('sigma', -1, 'Scale parameter, sigma value. -1 means adaptive learning')
flags.DEFINE_float('sigma_init', 1.0, 'Initial value of the scale parameter.')

## Local GAN flags
flags.DEFINE_string('data', ' ', 'LGAN domain name')
flags.DEFINE_integer('sz', 0, 'Experiment input size')
flags.DEFINE_integer('ch', 0, 'Experiment number of channels')
flags.DEFINE_string('lgan_nn', '', 'LGAN Network architecture')
flags.DEFINE_integer('ngf', 0, 'LGAN generator layers depth')
flags.DEFINE_integer('ndf', 0, 'LGAN discriminator layers depth')
flags.DEFINE_integer('nz', 0, 'Dimension of latent z vector')
flags.DEFINE_integer('src_nz', 0, 'Dimension of latent z vector for the source LGAN')
flags.DEFINE_integer('trg_nz', 0, 'Dimension of latent z vector for the target LGAN')
flags.DEFINE_integer('jcb', 0, 'Dimension of jacobian')
flags.DEFINE_float('alpha', 0, 'Weight for locality')
flags.DEFINE_float('beta', 0, 'Weight for orthogonality')
flags.DEFINE_float('theta', 0.1, 'Weight to enforce locality at z = 0')
flags.DEFINE_float('delta', 1e-4, 'Jacobian step size')
flags.DEFINE_float('lgan_var', 0, 'Gaussian noise variance for the training')

## Others
flags.DEFINE_string('gpu', '0', 'GPU number')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.adpt == 0: FLAGS.adpt_val = 0
    if FLAGS.src == 'mnist' and FLAGS.trg == 'mnistm': FLAGS.src_inv = True

    if FLAGS.sw == 0: FLAGS.sv = 0.0
    if FLAGS.tw == 0: FLAGS.tv = 0.0

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
        f"dw_{FLAGS.dw}",
        f"lw_{FLAGS.lw}",
        f"sw_{FLAGS.sw}",
        f"xw_{FLAGS.xw}",
        f"tw_{FLAGS.tw}",
        # f"cgw_{FLAGS.cgw}",
        f"cgw_{FLAGS.cgw}_{FLAGS.grad_val}",
        f"lpc_{FLAGS.lpc}",
        f"adpt_{FLAGS.adpt}_{FLAGS.adpt_val}",
        f"var_{FLAGS.sv}_{FLAGS.tv}"
    ]
    model_name = '_'.join(setup_list)
    print(colored(f"Model name: {model_name}", 'green'))

    if FLAGS.src in ['amazon', 'dslr', 'webcam', 'c', 'i', 'p']:
        exp_sz, exp_ch, _ = get_attr_image(FLAGS.nn, FLAGS.src)
        FLAGS.bs = 36
    else:
        _, _, exp_sz, _, _, exp_ch, _ = get_attr(FLAGS.src, FLAGS.trg)
    FLAGS.sz = exp_sz
    FLAGS.ch = exp_ch
    lgan_source, lgan_target = get_lgan_name(FLAGS.src, FLAGS.trg, exp_sz, exp_ch)

    # Make LGAN models and restore
    if FLAGS.sw > 0:
        update_lgan_flags(FLAGS.src, FLAGS)
        FLAGS.src_nz = FLAGS.nz
        Ls = lgan(FLAGS, inference=True)
        Ls.sess.run(tf.global_variables_initializer())
        var_lgan = tf.get_collection('trainable_variables', FLAGS.src + '/lgan/gen')
        path = tf.train.latest_checkpoint(os.path.join(FLAGS.ckptdir, lgan_source))
        tf.train.Saver(var_lgan).restore(Ls.sess, path)
        print(colored(f"Source LGAN model is restored from {path}", 'green'))
    else:
        Ls = None

    if FLAGS.tw > 0:
        update_lgan_flags(FLAGS.trg, FLAGS)
        FLAGS.trg_nz = FLAGS.nz
        Lt = lgan(FLAGS, inference=True)
        Lt.sess.run(tf.global_variables_initializer())
        var_lgan = tf.get_collection('trainable_variables', FLAGS.trg + '/lgan/gen')
        path = tf.train.latest_checkpoint(os.path.join(FLAGS.ckptdir, lgan_target))
        tf.train.Saver(var_lgan).restore(Lt.sess, path)
        print(colored(f"Target LGAN model is restored from {path}", 'green'))
    else:
        Lt = None

    # Make main model and initialize
    if FLAGS.src in ['amazon', 'dslr', 'webcam', 'c', 'i', 'p']:
        M = lpda_image(FLAGS)
    else:
        M = lpda(FLAGS)
    M.sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # Train the main model
    if FLAGS.src in ['amazon', 'dslr', 'webcam', 'c', 'i', 'p']:
        if FLAGS.nn == 'resnet':
            var_resnet = tf.get_collection('trainable_variables', 'resnet_model')
            var_resnet = [v for v in var_resnet if 'dense' not in v.name]
            path = tf.train.latest_checkpoint(os.path.join(FLAGS.datadir, 'resnet_imagenet_v2_fp32_20181001'))
            tf.train.Saver(var_resnet).restore(M.sess, path)
            print(colored(f"Resnet model is restored from {path}", 'green'))
        train_image(M, FLAGS, Ls, Lt, saver=saver, model_name=model_name)
    else:
        train(M, FLAGS, Ls, Lt, saver=saver, model_name=model_name)


if __name__ == '__main__':
    tf.app.run()
