import tensorflow as tf
from tensorbayes.layers import conv2d, conv2d_transpose, batch_norm
from tensorflow.contrib.framework import arg_scope

from utils import leaky_relu, relu, tanh, sigmoid


class lgan_small(object):
    def __init__(self, FLAGS):
        # self.FLAGS = FLAGS
        self.data = FLAGS.data
        self.ngf = FLAGS.ngf
        self.ndf = FLAGS.ndf
        self.nz = FLAGS.nz
        self.exp_sz = FLAGS.sz
        self.exp_ch = FLAGS.ch

    def generator(self, x, z, phase, enc_phase=False, dec_phase=False, reuse=tf.AUTO_REUSE, internal_update=False,
                  getter=None):
        with tf.variable_scope(self.data + '/lgan/gen', reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.2), \
                 arg_scope([conv2d], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([conv2d_transpose], activation=relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                layout = [
                    (conv2d, (self.ngf, 4, 2), dict(bn=False)),
                    (conv2d, (self.ngf * 2, 4, 2), {}),
                    (conv2d, (self.ngf * 4, 4, 2), {}),
                    (conv2d, (self.ngf * 8, 4, 2), {}),
                    (conv2d, (self.nz, 4, 2), dict(activation=None)),
                    (leaky_relu, (), {}),
                    (conv2d_transpose, (self.ngf * 8, 4, 2), {}),
                    (conv2d_transpose, (self.ngf * 4, 4, 2), {}),
                    (conv2d_transpose, (self.ngf * 2, 4, 2), {}),
                    (conv2d_transpose, (self.ngf, 4, 2), {}),
                    (conv2d_transpose, (self.exp_ch, 4, 2), dict(activation=tanh, bn=False))
                ]

                start = 0
                z_layer = 5
                end = len(layout)

                if enc_phase:
                    for i in range(start, z_layer):
                        with tf.variable_scope('l{:d}'.format(i)):
                            f, f_args, f_kwargs = layout[i]
                            x = f(x, *f_args, **f_kwargs)
                    return x

                elif dec_phase:
                    x = tf.expand_dims(tf.expand_dims(x, 1), 1)
                    for i in range(z_layer, end):
                        with tf.variable_scope('l{:d}'.format(i)):
                            f, f_args, f_kwargs = layout[i]
                            x = f(x, *f_args, **f_kwargs)

                elif not enc_phase and not dec_phase:
                    for i in range(start, z_layer):
                        with tf.variable_scope('l{:d}'.format(i)):
                            f, f_args, f_kwargs = layout[i]
                            x = f(x, *f_args, **f_kwargs)

                    x = x + tf.expand_dims(tf.expand_dims(z, 1), 1)

                    for i in range(z_layer, end):
                        with tf.variable_scope('l{:d}'.format(i)):
                            f, f_args, f_kwargs = layout[i]
                            x = f(x, *f_args, **f_kwargs)

                else:
                    raise ValueError("Network error")

        if self.exp_sz != 32:
            x = tf.image.resize_images(x, [self.exp_sz, self.exp_sz])

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
                    # (conv2d, (1, 4, 1), dict(activation=None, bn=False, padding='VALID'))
                ]

                for i in range(0, len(layout)):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x
