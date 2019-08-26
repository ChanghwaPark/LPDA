import tensorflow as tf
from tensorbayes.layers import conv2d, conv2d_transpose, batch_norm, dense
from tensorflow.contrib.framework import arg_scope

from utils import leaky_relu, relu, tanh, sigmoid, reshape, global_pool


class lgan_large(object):
    def __init__(self, FLAGS):
        # self.FLAGS = FLAGS
        self.data = FLAGS.data
        self.ngf = FLAGS.ngf
        self.ndf = FLAGS.ndf
        self.nz = FLAGS.nz
        self.exp_sz = FLAGS.sz
        self.exp_ch = FLAGS.ch

    def generator(self, x, z, phase, reuse=tf.AUTO_REUSE, internal_update=False, getter=None):
        with tf.variable_scope(self.data + '/lgan/gen', reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.2), \
                 arg_scope([dense], activation=None, bn=True, phase=phase), \
                 arg_scope([conv2d], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([conv2d_transpose], activation=relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                layout = [
                    (conv2d, (self.ngf, 5, 2), dict(bn=False)),
                    (conv2d, (self.ngf * 2, 5, 2), {}),
                    (conv2d, (self.ngf * 4, 5, 2), {}),
                    (global_pool, (), dict(axis=[1, 2], keepdims=True)),
                    (dense, (), dict(num_outputs=8192)),
                    (reshape, (), dict(shape=[-1, 4, 4, 512])),
                    (conv2d, (512, 3, 1), {}),
                    (conv2d, (512, 1, 1), {}),
                    (conv2d_transpose, (256, 5, 2), {}),
                    (conv2d, (256, 3, 1), {}),
                    (conv2d, (256, 1, 1), {}),
                    (conv2d_transpose, (128, 5, 2), {}),
                    (conv2d, (128, 3, 1), {}),
                    (conv2d, (128, 1, 1), {}),
                    (conv2d_transpose, (self.exp_ch, 5, 2), dict(activation=tanh, bn=False))
                ]

                start = 0
                z_layer = 4
                end = len(layout)

                for i in range(start, z_layer):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

                x = tf.concat([x, tf.expand_dims(tf.expand_dims(z, 1), 1)], 3)

                for i in range(z_layer, end):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x

    def discriminator(self, x, phase, reuse=tf.AUTO_REUSE, internal_update=False, getter=None):
        with tf.variable_scope(self.data + '/lgan/dsc', reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.2), \
                 arg_scope([conv2d], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                layout = [
                    (conv2d, (self.ndf, 4, 2), dict(bn=False)),
                    (conv2d, (self.ndf * 2, 4, 2), {}),
                    (conv2d, (self.ndf * 4, 4, 2), {}),
                    (conv2d, (1, 4, 1), dict(activation=sigmoid, bn=False, padding='VALID'))
                ]

                for i in range(0, len(layout)):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x
