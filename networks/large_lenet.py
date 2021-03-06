import tensorflow as tf
from tensorbayes.layers import dense, conv2d, max_pool, batch_norm, instance_norm
from tensorflow.contrib.framework import arg_scope
from utils import relu
from data.dataset import get_attr


class large_lenet(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        _, _, _, _, _, _, self.nc = get_attr(FLAGS.src, FLAGS.trg)

    def get_flen(self):
        if self.FLAGS.trim == 0:
            return self.nc
        elif self.FLAGS.trim == 1:
            return 2048
        elif self.FLAGS.trim == 2:
            return 3072
        else:
            raise ValueError('Trim for the lenet network should be 0 or 1')

    def classifier(self, x, phase, enc_phase=True, trim=0, scope='class', reuse=tf.AUTO_REUSE, internal_update=False,
                   getter=None):
        with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
            with arg_scope([conv2d], activation=relu, bn=True, phase=phase), \
                 arg_scope([dense], activation=relu, bn=False, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                preprocess = instance_norm if self.FLAGS.inorm else tf.identity
                layout = [
                    (preprocess, (), {}),
                    (conv2d, (64, 5, 1), {}),
                    (max_pool, (3, 2), {}),
                    (conv2d, (64, 5, 1), {}),
                    (max_pool, (3, 2), {}),
                    (conv2d, (128, 5, 1), {}),
                    (dense, (3072,), {}),
                    (dense, (2048,), {}),
                    (dense, (self.nc,), dict(activation=None))
                ]

                if enc_phase:
                    start = 0
                    end = len(layout) - trim
                else:
                    start = len(layout) - trim
                    end = len(layout)

                for i in range(start, end):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x

    def feature_discriminator(self, x, phase, C=1, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('disc/feat', reuse=reuse):
            with arg_scope([dense], activation=relu, bn=False):
                x = dense(x, 1024)
                x = dense(x, 1024)
                x = dense(x, C, activation=None)

        return x
